"""
Phase 3 — Skin displacement simulation for orthognathic surgery.

Approach: prescribed-displacement boundary conditions + Laplacian propagation on
the skin surface mesh.
  - No gmsh / FEBio required (works directly on the surface mesh)
  - The surgical plan is a rigid transform of the mandible (translation + pitch)
  - Each jaw-adjacent skin node is pushed by its nearest bone node's motion,
    scaled by a region-dependent soft-tissue-to-hard-tissue ratio
  - Skull-vault nodes are fixed (zero displacement)
  - Iterative Laplacian smoothing propagates displacement to all other nodes

Two things this fixes over the first version:
  1. Axes are derived from the patient's anatomy (pipeline/anatomy.py), not
     assumed. The old code pushed along mesh axis 1 and anchored on axis 2, which
     for this scanner's export meant "move the jaw backwards" and "fix the left
     side of the head" — a confident simulation of the wrong operation.
  2. Jaw-adjacent skin no longer moves 1:1 with bone everywhere. Soft tissue
     follows bone by a ratio that falls off from chin to lip to cheek.

CAVEAT: the ratios below are population means from the orthognathic literature,
not patient-specific measurements. Until Phase 5 has a paired post-op scan, the
output is a calibrated estimate, not a validated prediction.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.anatomy import AnatomicalFrame, detect_frame
from pipeline.phases.base import Phase

# ── surgical scenario (env-overridable so the web UI can drive it) ────────────
#   FACESIM_ADVANCE_MM   mandible advancement, anterior +  (setback = negative)
#   FACESIM_VERTICAL_MM  vertical movement, superior +
#   FACESIM_LATERAL_MM   asymmetry correction, patient-left +
#   FACESIM_PITCH_DEG    rotation about the lateral axis through the mandible
#                        centroid; positive = chin rotates anteriorly (counter-
#                        clockwise seen from the patient's right side)
ADVANCE_MM = float(os.getenv("FACESIM_ADVANCE_MM", "5.0"))
VERTICAL_MM = float(os.getenv("FACESIM_VERTICAL_MM", "0.0"))
LATERAL_MM = float(os.getenv("FACESIM_LATERAL_MM", "0.0"))
PITCH_DEG = float(os.getenv("FACESIM_PITCH_DEG", "0.0"))

JAW_RADIUS_MM = float(os.getenv("FACESIM_JAW_RADIUS_MM", "15.0"))
SKULL_PERCENTILE = float(os.getenv("FACESIM_SKULL_PERCENTILE", "95"))
N_ITER = int(os.getenv("FACESIM_ITERS", "300"))

# ── soft-tissue-to-hard-tissue response ──────────────────────────────────────
# Keys are the normalized height of a skin node over the mandible's superior
# extent: 0.0 = menton (chin bottom), 1.0 = top of the ramus/condyle.
# Values are sagittal soft:hard movement ratios for mandibular advancement, taken
# as mid-range population means from orthognathic outcome studies:
#   chin / soft pogonion  ~0.9–1.0
#   B point / labiomental ~0.8–0.9
#   lower lip             ~0.6–0.75
#   cheek / masseter      falls off to ~0 — mandibular surgery barely moves it
# Override with FACESIM_RATIO_PROFILE="0.0:0.95,0.5:0.7,1.0:0.05".
DEFAULT_RATIO_PROFILE = (
    (0.00, 0.95),   # submental / menton
    (0.20, 0.95),   # pogonion
    (0.35, 0.90),   # B point, labiomental fold
    (0.50, 0.70),   # lower lip
    (0.65, 0.35),   # oral commissure
    (0.80, 0.15),   # cheek over the ramus
    (1.00, 0.05),   # preauricular
)


def _parse_ratio_profile(raw: str) -> tuple[tuple[float, float], ...]:
    pairs = []
    for chunk in raw.split(","):
        t, r = chunk.split(":")
        pairs.append((float(t), float(r)))
    return tuple(sorted(pairs))


RATIO_PROFILE = (
    _parse_ratio_profile(os.environ["FACESIM_RATIO_PROFILE"])
    if os.getenv("FACESIM_RATIO_PROFILE")
    else DEFAULT_RATIO_PROFILE
)


@dataclass(frozen=True)
class Scenario:
    """One surgical plan. The CLI pipeline builds it from the environment; the
    web server builds one per request, which is why this is a value passed in
    rather than module state."""

    advance_mm: float = ADVANCE_MM
    vertical_mm: float = VERTICAL_MM
    lateral_mm: float = LATERAL_MM
    pitch_deg: float = PITCH_DEG
    jaw_radius_mm: float = JAW_RADIUS_MM
    skull_percentile: float = SKULL_PERCENTILE
    n_iter: int = N_ITER
    ratio_profile: tuple[tuple[float, float], ...] = RATIO_PROFILE

    @classmethod
    def from_env(cls) -> "Scenario":
        return cls()

    def to_dict(self) -> dict:
        return {
            "advance_mm": self.advance_mm,
            "vertical_mm": self.vertical_mm,
            "lateral_mm": self.lateral_mm,
            "pitch_deg": self.pitch_deg,
        }


class Phase3Sim(Phase):
    name = "phase3_simulation"

    def __init__(self, scenario: Scenario | None = None):
        self.scenario = scenario or Scenario.from_env()

    def is_available(self) -> tuple[bool, str]:
        return True, ""  # no external tools required

    def artifacts_exist(self, data_dir: Path) -> bool:
        return (data_dir / "sim" / "deformed.vtk").exists()

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state, data_dir: Path) -> dict:
        import trimesh

        sim_dir = data_dir / "sim"
        sim_dir.mkdir(exist_ok=True)

        stl_dir = data_dir / "stl"
        jaw_stl = stl_dir / "teeth_lower_jawbone.stl"
        skin_stl = stl_dir / "soft_skin.stl"
        for p in (jaw_stl, skin_stl):
            if not p.exists():
                raise FileNotFoundError(f"Required STL not found: {p}")

        frame = detect_frame(stl_dir)
        print(f"[phase3] anatomical frame ({frame.source})")
        print(f"[phase3]   superior={np.round(frame.superior, 3)}"
              f" anterior={np.round(frame.anterior, 3)} lateral={np.round(frame.lateral, 3)}")

        skin = trimesh.load(str(skin_stl), force="mesh")
        verts = np.array(skin.vertices, dtype=float)
        faces = np.array(skin.faces, dtype=int)
        print(f"[phase3] skin mesh: {len(verts)} verts, {len(faces)} faces")

        jaw = trimesh.load(str(jaw_stl), force="mesh")
        jaw_verts = np.array(jaw.vertices, dtype=float)

        sc = self.scenario
        print(f"[phase3] scenario: advance={sc.advance_mm}mm vertical={sc.vertical_mm}mm "
              f"lateral={sc.lateral_mm}mm pitch={sc.pitch_deg}deg")
        jaw_disp = self._rigid_displacement(jaw_verts, frame)
        print(f"[phase3] mandible motion: max {np.linalg.norm(jaw_disp, axis=1).max():.2f} mm")

        skull_ids, jaw_ids, free_ids, prescribed = self._boundary_conditions(
            verts, jaw_verts, jaw_disp, frame
        )
        print(f"[phase3] skull fixed: {len(skull_ids)}  jaw-driven: {len(jaw_ids)}  free: {len(free_ids)}")

        if len(jaw_ids) == 0:
            raise RuntimeError(
                f"No skin nodes within {sc.jaw_radius_mm}mm of jaw — "
                "STL coordinate systems may not align."
            )

        print(f"[phase3] running Laplacian propagation ({sc.n_iter} iterations)...")
        deformed_verts = self._propagate(verts, faces, skull_ids, jaw_ids, free_ids, prescribed)

        disp_mag = np.linalg.norm(deformed_verts - verts, axis=1)
        max_disp = float(disp_mag.max())
        print(f"[phase3] max skin displacement: {max_disp:.2f} mm")

        deformed_vtk = sim_dir / "deformed.vtk"
        self._save_vtk(deformed_verts, faces, deformed_vtk)

        before_dir, after_dir = self._save_scene_stls(
            verts, faces, deformed_verts, jaw_stl, jaw_verts, jaw_disp, data_dir
        )

        return {
            "deformed_vtk": str(deformed_vtk),
            "before": str(before_dir),
            "after": str(after_dir),
            "after_dir": str(after_dir),
            "n_verts": len(verts),
            "max_disp_mm": round(max_disp, 3),
            "mean_disp_mm": round(float(disp_mag.mean()), 3),
            "scenario": sc.to_dict(),
            "frame": frame.to_dict(),
            "ratio_profile": [list(p) for p in sc.ratio_profile],
            "method": "prescribed_ratio+laplacian",
        }

    def run_fallback(self, state, data_dir: Path) -> dict:
        p2 = state.phase("phase2_meshing")
        after = p2.get("artifacts", {}).get("after")
        if after and Path(after).exists():
            print("[phase3] using Phase 2 proxy meshes as fallback")
            return {"method": "phase2_proxy", "after_dir": after}
        raise NotImplementedError("Phase 2 artifacts not available for fallback")

    # ── surgical plan → bone displacement field ───────────────────────

    def _rigid_displacement(self, jaw_verts: np.ndarray, frame: AnatomicalFrame) -> np.ndarray:
        """Displacement of every mandible vertex under the planned rigid motion.

        Rotation is applied about the lateral axis through the mandible centroid
        (a stand-in for the condylar axis), then the translation is added."""
        sc = self.scenario
        translation = frame.project({
            "anterior": sc.advance_mm,
            "superior": sc.vertical_mm,
            "lateral": sc.lateral_mm,
        })

        if abs(sc.pitch_deg) < 1e-9:
            return np.broadcast_to(translation, jaw_verts.shape).copy()

        axis = frame.lateral
        theta = np.radians(sc.pitch_deg)
        pivot = jaw_verts.mean(axis=0)
        rel = jaw_verts - pivot
        # Rodrigues' rotation formula, vectorized over all vertices
        rotated = (
            rel * np.cos(theta)
            + np.cross(axis, rel) * np.sin(theta)
            + np.outer(rel @ axis, axis) * (1.0 - np.cos(theta))
        )
        return (rotated + pivot + translation) - jaw_verts

    # ── boundary conditions on the skin mesh ──────────────────────────

    def _boundary_conditions(
        self,
        verts: np.ndarray,
        jaw_verts: np.ndarray,
        jaw_disp: np.ndarray,
        frame: AnatomicalFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (skull_ids, jaw_ids, free_ids, prescribed_disp_for_jaw_ids)."""
        from scipy.spatial import cKDTree

        # Skull vault: the top slab along the TRUE superior axis, held fixed.
        s_skin = verts @ frame.superior
        sc = self.scenario
        skull_mask = s_skin >= np.percentile(s_skin, sc.skull_percentile)

        dists, nearest = cKDTree(jaw_verts).query(verts)
        jaw_mask = (dists < sc.jaw_radius_mm) & ~skull_mask

        jaw_ids = np.where(jaw_mask)[0]
        skull_ids = np.where(skull_mask)[0]
        free_ids = np.where(~jaw_mask & ~skull_mask)[0]

        # Region-dependent soft:hard ratio, by height over the mandible extent.
        s_jaw = jaw_verts @ frame.superior
        lo, hi = float(s_jaw.min()), float(s_jaw.max())
        span = max(hi - lo, 1e-6)
        t = np.clip((s_skin[jaw_ids] - lo) / span, 0.0, 1.0)
        profile = sc.ratio_profile
        ratios = np.interp(t, [p[0] for p in profile], [p[1] for p in profile])

        prescribed = jaw_disp[nearest[jaw_ids]] * ratios[:, None]
        print(f"[phase3] soft:hard ratio over jaw-driven nodes — "
              f"min {ratios.min():.2f} / mean {ratios.mean():.2f} / max {ratios.max():.2f}")
        return skull_ids, jaw_ids, free_ids, prescribed

    # ── Laplacian propagation ─────────────────────────────────────────

    def _propagate(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        skull_ids: np.ndarray,
        jaw_ids: np.ndarray,
        free_ids: np.ndarray,
        prescribed: np.ndarray,
    ) -> np.ndarray:
        import time
        from scipy.sparse import coo_matrix

        n = len(verts)
        print(f"[phase3] building sparse Laplacian ({len(free_ids)} free nodes)...")
        t0 = time.time()
        # Build all directed edges from face array — fully vectorized, no Python loop
        edges = np.vstack([
            faces[:, [0, 1]], faces[:, [1, 0]],
            faces[:, [1, 2]], faces[:, [2, 1]],
            faces[:, [0, 2]], faces[:, [2, 0]],
        ])
        free_mask = np.zeros(n, dtype=bool)
        free_mask[free_ids] = True
        edges = edges[free_mask[edges[:, 0]]]   # keep only free-node sources
        src, dst = edges[:, 0], edges[:, 1]
        degree = np.bincount(src, minlength=n).astype(float)
        degree[degree == 0] = 1.0
        W = coo_matrix((1.0 / degree[src], (src, dst)), shape=(n, n)).tocsr()
        print(f"[phase3] sparse matrix built in {time.time()-t0:.1f}s")

        disp = np.zeros_like(verts)
        disp[jaw_ids] = prescribed

        t0 = time.time()
        n_iter = self.scenario.n_iter
        for iteration in range(n_iter):
            disp = W @ disp
            disp[jaw_ids] = prescribed      # Dirichlet BC: bone-driven skin
            disp[skull_ids] = 0.0           # Dirichlet BC: fixed skull vault
            if iteration % 10 == 9:
                elapsed = time.time() - t0
                eta = elapsed / (iteration + 1) * (n_iter - iteration - 1)
                print(f"[phase3]   iter {iteration+1}/{n_iter}  elapsed={elapsed:.1f}s  eta={eta:.0f}s")

        return verts + disp

    # ── save outputs ──────────────────────────────────────────────────

    def _save_vtk(self, verts: np.ndarray, faces: np.ndarray, out: Path):
        import meshio
        meshio.write(str(out), meshio.Mesh(points=verts, cells=[("triangle", faces)]))

    def _save_scene_stls(
        self,
        orig_verts: np.ndarray,
        faces: np.ndarray,
        deformed_verts: np.ndarray,
        jaw_stl: Path,
        jaw_verts: np.ndarray,
        jaw_disp: np.ndarray,
        data_dir: Path,
    ) -> tuple[Path, Path]:
        import trimesh

        before_dir = data_dir / "mesh" / "before"
        after_dir = data_dir / "mesh" / "after"
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)

        trimesh.Trimesh(orig_verts, faces).export(str(before_dir / "skin.stl"))
        trimesh.Trimesh(deformed_verts, faces).export(str(after_dir / "skin.stl"))

        # The "after" jaw is the operated jaw — exporting the pre-op mesh here made
        # the render show the surgery moving skin but not bone.
        shutil.copy(jaw_stl, before_dir / "jaw.stl")
        jaw_mesh = trimesh.load(str(jaw_stl), force="mesh")
        trimesh.Trimesh(jaw_verts + jaw_disp, np.array(jaw_mesh.faces, dtype=int)).export(
            str(after_dir / "jaw.stl")
        )

        return before_dir, after_dir
