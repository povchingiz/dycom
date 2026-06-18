from pathlib import Path
from pipeline.phases.base import Phase


class Phase5Validate(Phase):
    name = "phase5_validation"

    def artifacts_exist(self, data_dir: Path) -> bool:
        return (data_dir / "validation" / "metrics.json").exists()

    def run(self, state, data_dir: Path) -> dict:
        post_op_dir = data_dir / "post_op"
        candidates = list(post_op_dir.glob("*.dcm")) + list(post_op_dir.glob("*.nii.gz")) if post_op_dir.exists() else []
        if not candidates:
            state.mark(
                self.name, "waiting",
                reason="need post-op scan — place in data/post_op/ (DCM or NIfTI)"
            )
            return {}

        # Surface distance: predicted after vs real post-op
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        p3 = state.phase("phase3_simulation")
        after_dir = Path(p3.get("artifacts", {}).get("after_dir", ""))
        pred_skin = after_dir / "skin.stl"
        if not pred_skin.exists():
            raise FileNotFoundError("Predicted skin mesh not found — run Phase 3 first")

        # TODO: segment post-op scan → real_skin.stl
        # For now: placeholder
        val_dir = data_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        import json
        metrics = {"status": "placeholder", "note": "segment post_op scan to get real_skin.stl"}
        (val_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return {"metrics": str(val_dir / "metrics.json")}
