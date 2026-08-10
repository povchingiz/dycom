"""
Phase 5 — Validation of the predicted face against a real post-op scan.

The pipeline's only real claim is "this is what the face will look like". Dice on
bone says nothing about it. This phase measures the claim directly:

  1. segment the post-op CBCT the same way the pre-op one was segmented
  2. rigidly register it to the pre-op scan using the cranial vault — the one
     region a mandibular osteotomy cannot move
  3. measure surface distance from the PREDICTED post-op face to the REAL one
  4. measure the same distance from the untouched PRE-OP face to the real one

Step 4 is the point. It is the do-nothing control: if predicting the surgery is
no closer to the truth than ignoring it, the simulation adds nothing, and only
this comparison can tell you. Reported as `improvement_pct`.

Distances are reported over the whole face and over the lower third (the region
the surgery actually moves), plus the fraction within 2 mm — the tolerance
usually quoted for clinically useful soft-tissue prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.phases.base import Phase

# Fraction of the head height (from the top) used as the rigid-registration
# anchor. The cranial vault does not move during mandibular surgery.
ANCHOR_TOP_FRACTION = 0.35
# Points sampled on each surface for the distance metrics. Point-to-surface
# queries are exact but O(n log n) with a large constant; 50k points already
# resolves sub-millimetre differences on a face.
MAX_SAMPLE_POINTS = 50_000
CLINICAL_TOLERANCE_MM = 2.0


class Phase5Validate(Phase):
    name = "phase5_validation"

    def artifacts_exist(self, data_dir: Path) -> bool:
        metrics = data_dir / "validation" / "metrics.json"
        if not metrics.exists():
            return False
        try:
            return json.loads(metrics.read_text()).get("status") == "validated"
        except (json.JSONDecodeError, OSError):
            return False

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state, data_dir: Path) -> dict:
        post_op_dir = data_dir / "post_op"
        candidates = (
            sorted(post_op_dir.glob("*.nii.gz")) + sorted(post_op_dir.glob("*.dcm"))
            if post_op_dir.exists() else []
        )
        if not candidates:
            state.mark(
                self.name, "waiting",
                reason="need post-op scan — place in data/post_op/ (DCM or NIfTI)"
            )
            return {}

        import trimesh

        pred_skin_path = self._predicted_skin(state, data_dir)
        pre_skin_path = data_dir / "stl" / "soft_skin.stl"
        if not pre_skin_path.exists():
            raise FileNotFoundError(f"Pre-op skin mesh not found: {pre_skin_path}")

        val_dir = data_dir / "validation"
        val_dir.mkdir(exist_ok=True)

        real_skin_path = self._segment_post_op(candidates[0], val_dir)

        pred = trimesh.load(str(pred_skin_path), force="mesh")
        pre = trimesh.load(str(pre_skin_path), force="mesh")
        real = trimesh.load(str(real_skin_path), force="mesh")

        # Post-op scan sits in its own scanner frame; align it to the pre-op one.
        transform, fit_mm = self._register(real, pre, data_dir)
        real.apply_transform(transform)
        real.export(str(val_dir / "real_skin_registered.stl"))
        print(f"[phase5] rigid registration residual: {fit_mm:.3f} mm on the cranial anchor")

        roi_mask = self._lower_third_mask(real, data_dir)

        metrics = {
            "status": "validated",
            "post_op_source": str(candidates[0]),
            "registration_residual_mm": round(fit_mm, 4),
            "predicted_vs_real": self._surface_metrics(pred, real, roi_mask),
            "preop_vs_real": self._surface_metrics(pre, real, roi_mask),
        }
        metrics["improvement_pct"] = self._improvement(metrics)
        metrics["verdict"] = self._verdict(metrics)

        (val_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        self._print_report(metrics)
        return {"metrics": str(val_dir / "metrics.json"),
                "improvement_pct": metrics["improvement_pct"],
                "verdict": metrics["verdict"]}

    # ── inputs ────────────────────────────────────────────────────────

    @staticmethod
    def _predicted_skin(state, data_dir: Path) -> Path:
        after_dir = Path(
            state.phase("phase3_simulation").get("artifacts", {}).get("after_dir", "")
        ) if state is not None else Path()
        if not (after_dir / "skin.stl").exists():
            after_dir = data_dir / "mesh" / "after"
        pred = after_dir / "skin.stl"
        if not pred.exists():
            raise FileNotFoundError("Predicted skin mesh not found — run Phase 3 first")
        return pred

    @staticmethod
    def _segment_post_op(scan: Path, val_dir: Path) -> Path:
        """Segment the post-op scan with the same code path as the pre-op one —
        a different segmentation would make the comparison measure the
        segmenter, not the simulation."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from segmentation.dcm_to_nifti import convert
        from segmentation.masks_to_stl import nifti_to_stl
        from segmentation.segment_soft_tissue import segment

        out_stl = val_dir / "real_skin.stl"
        if out_stl.exists():
            print(f"[phase5] reusing {out_stl}")
            return out_stl

        nifti = scan
        if scan.suffix == ".dcm":
            nifti = val_dir / "post_op.nii.gz"
            if not nifti.exists():
                print(f"[phase5] converting {scan.name} → NIfTI")
                convert(str(scan), str(nifti))

        print(f"[phase5] segmenting post-op soft tissue from {nifti.name}")
        seg_dir = val_dir / "seg"
        segment(str(nifti), str(seg_dir))
        nifti_to_stl(seg_dir / "skin.nii.gz", out_stl)
        return out_stl

    # ── registration ──────────────────────────────────────────────────

    def _register(self, moving, fixed, data_dir: Path) -> tuple[np.ndarray, float]:
        """Rigid alignment driven by the unoperated part of the face.

        The anchor is everything above the mandible — forehead, orbits, nose,
        zygomas. That is the standard superimposition region for orthognathic
        follow-up: mandibular surgery does not move it, and unlike the bare
        cranial vault it has enough shape that ICP cannot slide tangentially.
        Registering on the whole face instead would drag the anchor toward the
        operated region and hide exactly the error being measured."""
        import trimesh

        up, cut = self._anchor_plane(fixed, data_dir)

        src = np.asarray(moving.vertices)[self._above(np.asarray(moving.vertices), up, cut)]
        dst_mask = self._above(np.asarray(fixed.vertices), up, cut)
        dst_mesh = trimesh.Trimesh(*self._submesh(fixed, dst_mask), process=False)
        print(f"[phase5] registration anchor: {len(src)} moving pts → "
              f"{len(dst_mesh.faces)} fixed faces")

        # Centroid pre-alignment: ICP is local, and two scanner frames can differ
        # by centimetres. Without this it converges into a nearby wrong minimum.
        initial = np.eye(4)
        initial[:3, 3] = np.asarray(dst_mesh.vertices).mean(axis=0) - src.mean(axis=0)

        matrix, _, cost = trimesh.registration.icp(
            src, dst_mesh, initial=initial, scale=False,
            max_iterations=200, threshold=1e-6,
        )
        return matrix, float(np.sqrt(cost)) if cost is not None else float("nan")

    def _anchor_plane(self, fixed, data_dir: Path) -> tuple[np.ndarray, float]:
        """(up vector, cut height) separating the unoperated face from the jaw."""
        try:
            import trimesh
            from pipeline.anatomy import detect_frame
            up = detect_frame(data_dir / "stl").superior
            jaw = trimesh.load(str(data_dir / "stl" / "teeth_lower_jawbone.stl"), force="mesh")
            return up, float((np.asarray(jaw.vertices) @ up).max())
        except (RuntimeError, FileNotFoundError, ValueError):
            # No jaw STLs to derive the frame from — fall back to the mesh's own
            # long axis and a fixed top fraction.
            print("[phase5] anatomical frame unavailable — using mesh principal axis as 'up'")
            up = self._dominant_axis(fixed.vertices)
            s = np.asarray(fixed.vertices) @ up
            return up, float(np.percentile(s, 100 * (1 - ANCHOR_TOP_FRACTION)))

    @staticmethod
    def _above(verts: np.ndarray, up: np.ndarray, cut: float) -> np.ndarray:
        return (verts @ up) >= cut

    @staticmethod
    def _dominant_axis(verts: np.ndarray) -> np.ndarray:
        centered = np.asarray(verts) - np.asarray(verts).mean(axis=0)
        _, _, vh = np.linalg.svd(centered[:: max(1, len(centered) // 20000)], full_matrices=False)
        return vh[0]

    @staticmethod
    def _submesh(mesh, vertex_mask: np.ndarray):
        faces = np.asarray(mesh.faces)
        keep = vertex_mask[faces].all(axis=1)
        idx = np.where(vertex_mask)[0]
        remap = -np.ones(len(vertex_mask), dtype=int)
        remap[idx] = np.arange(len(idx))
        return np.asarray(mesh.vertices)[idx], remap[faces[keep]]

    # ── metrics ───────────────────────────────────────────────────────

    def _lower_third_mask(self, real, data_dir: Path) -> np.ndarray | None:
        """Vertices of the reference surface that lie over the mandible — the only
        region the operation is supposed to change."""
        try:
            from pipeline.anatomy import detect_frame
            import trimesh
            frame = detect_frame(data_dir / "stl")
            jaw = trimesh.load(str(data_dir / "stl" / "teeth_lower_jawbone.stl"), force="mesh")
        except (RuntimeError, FileNotFoundError, ValueError):
            return None
        s_jaw = np.asarray(jaw.vertices) @ frame.superior
        return (np.asarray(real.vertices) @ frame.superior) <= float(s_jaw.max())

    def _surface_metrics(self, query, reference, roi_mask: np.ndarray | None) -> dict:
        """Distance from every sampled point of `reference` to the `query` surface.

        Measured reference→query on purpose: it answers "how far is the real face
        from what we drew", which is the question a surgeon asks."""
        import trimesh

        pts = np.asarray(reference.vertices)
        idx = self._sample_indices(len(pts))
        _, dist, _ = trimesh.proximity.closest_point(query, pts[idx])
        dist = np.abs(dist)

        out = self._distance_stats(dist)
        if roi_mask is not None:
            roi = roi_mask[idx]
            if roi.any():
                out["lower_third"] = self._distance_stats(dist[roi])
        return out

    @staticmethod
    def _sample_indices(n: int) -> np.ndarray:
        if n <= MAX_SAMPLE_POINTS:
            return np.arange(n)
        # Deterministic stride, not a random draw — the number must be reproducible.
        return np.arange(0, n, int(np.ceil(n / MAX_SAMPLE_POINTS)))

    @staticmethod
    def _distance_stats(d: np.ndarray) -> dict:
        return {
            "n_points": int(d.size),
            "mean_mm": round(float(d.mean()), 4),
            "rms_mm": round(float(np.sqrt((d ** 2).mean())), 4),
            "median_mm": round(float(np.median(d)), 4),
            "p90_mm": round(float(np.percentile(d, 90)), 4),
            "p95_mm": round(float(np.percentile(d, 95)), 4),
            "max_mm": round(float(d.max()), 4),
            f"within_{CLINICAL_TOLERANCE_MM:g}mm_pct": round(
                100.0 * float((d <= CLINICAL_TOLERANCE_MM).mean()), 2),
        }

    @staticmethod
    def _improvement(metrics: dict) -> float | None:
        """How much closer the prediction is than doing nothing, in percent of the
        do-nothing error. Computed on the operated region when available."""
        def pick(block):
            return (block.get("lower_third") or block).get("rms_mm")

        base = pick(metrics["preop_vs_real"])
        pred = pick(metrics["predicted_vs_real"])
        if not base:
            return None
        return round(100.0 * (base - pred) / base, 2)

    @staticmethod
    def _verdict(metrics: dict) -> str:
        imp = metrics.get("improvement_pct")
        if imp is None:
            return "inconclusive"
        if imp <= 0:
            return "no better than doing nothing — simulation adds no accuracy"
        block = metrics["predicted_vs_real"].get("lower_third") or metrics["predicted_vs_real"]
        within = block.get(f"within_{CLINICAL_TOLERANCE_MM:g}mm_pct", 0.0)
        if within >= 90.0:
            return f"clinically useful: {within:.1f}% of the face within {CLINICAL_TOLERANCE_MM:g}mm"
        return f"better than baseline (+{imp:.1f}%) but only {within:.1f}% within {CLINICAL_TOLERANCE_MM:g}mm"

    @staticmethod
    def _print_report(m: dict) -> None:
        def line(tag, block):
            b = block.get("lower_third") or block
            print(f"[phase5]   {tag:<18} RMS {b['rms_mm']:>7.3f} mm   "
                  f"p95 {b['p95_mm']:>7.3f} mm   "
                  f"within 2mm {b[f'within_{CLINICAL_TOLERANCE_MM:g}mm_pct']:>6.2f}%")

        print("[phase5] surface distance to the real post-op face (lower third):")
        line("predicted", m["predicted_vs_real"])
        line("pre-op (baseline)", m["preop_vs_real"])
        print(f"[phase5] improvement over doing nothing: {m['improvement_pct']}%")
        print(f"[phase5] verdict: {m['verdict']}")
