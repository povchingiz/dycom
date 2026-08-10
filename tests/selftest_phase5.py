#!/usr/bin/env python
"""
Self-test for Phase 5 validation — checks the validator, not the simulation.

A validation phase that silently reports good numbers is worse than none, so it
gets its own ground truth: take the predicted face, move it into a different
"scanner frame" with a known rigid transform, and feed it back as if it were the
real post-op scan. A correct validator must undo the transform and report a
near-zero error. Anything else means the registration or the distance metric is
broken, and every future number it prints is meaningless.

Usage:
  python tests/selftest_phase5.py            # needs data/mesh/after/skin.stl (Phase 3)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phases.p5_validate import Phase5Validate  # noqa: E402

# Deliberately awkward offset: a few degrees about an oblique axis plus a
# centimetre-scale shift, i.e. what two scanner frames actually differ by.
ROTATION_DEG = 3.0
ROTATION_AXIS = (0.2, 1.0, 0.1)
TRANSLATION_MM = (7.0, -4.0, 2.5)

TOL_RESIDUAL_MM = 0.5   # registration must recover the transform this closely
TOL_RMS_MM = 0.5        # a perfect prediction must score near zero


def main(data_dir: Path = Path("data")) -> int:
    pred_path = data_dir / "mesh" / "after" / "skin.stl"
    if not pred_path.exists():
        print(f"FAIL: {pred_path} missing — run Phase 3 first")
        return 1

    val_dir = data_dir / "validation"
    post_op_dir = data_dir / "post_op"
    created = [p for p in (val_dir, post_op_dir) if not p.exists()]
    val_dir.mkdir(parents=True, exist_ok=True)
    post_op_dir.mkdir(parents=True, exist_ok=True)

    real_path = val_dir / "real_skin.stl"
    if real_path.exists():
        print(f"FAIL: {real_path} already exists — refusing to overwrite real data")
        return 1

    try:
        pred = trimesh.load(str(pred_path), force="mesh")
        matrix = trimesh.transformations.rotation_matrix(
            np.radians(ROTATION_DEG), ROTATION_AXIS, pred.centroid)
        matrix[:3, 3] += np.asarray(TRANSLATION_MM)

        real = pred.copy()
        real.apply_transform(matrix)
        real.export(str(real_path))
        marker = post_op_dir / "selftest.nii.gz"
        marker.touch()

        print(f"synthetic post-op: {ROTATION_DEG}deg about {ROTATION_AXIS} + {TRANSLATION_MM} mm")
        result = Phase5Validate().run(None, data_dir)

        import json
        metrics = json.loads((val_dir / "metrics.json").read_text())
        residual = metrics["registration_residual_mm"]
        block = metrics["predicted_vs_real"].get("lower_third") or metrics["predicted_vs_real"]
        rms = block["rms_mm"]

        ok = True
        for label, value, tol in (("registration residual", residual, TOL_RESIDUAL_MM),
                                  ("predicted-vs-real RMS", rms, TOL_RMS_MM)):
            status = "ok" if value <= tol else "FAIL"
            ok &= value <= tol
            print(f"  {status}: {label} = {value:.4f} mm (tolerance {tol} mm)")

        if metrics["predicted_vs_real"] == metrics["preop_vs_real"]:
            print("  FAIL: prediction and baseline scored identically — control is not wired up")
            ok = False

        print("PASS" if ok else "FAIL", "-", result.get("verdict"))
        return 0 if ok else 1
    finally:
        # Never leave synthetic ground truth lying around where a later run could
        # mistake it for a genuine post-op scan.
        for path in (real_path, val_dir / "metrics.json", val_dir / "real_skin_registered.stl",
                     post_op_dir / "selftest.nii.gz"):
            path.unlink(missing_ok=True)
        for path in created:
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
