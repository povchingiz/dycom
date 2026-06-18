from pathlib import Path
from pipeline.phases.base import Phase


class Phase1Seg(Phase):
    name = "phase1_segmentation"

    def artifacts_exist(self, data_dir: Path) -> bool:
        stl_dir = data_dir / "stl"
        return stl_dir.exists() and len(list(stl_dir.glob("*.stl"))) >= 10

    def run(self, state, data_dir: Path) -> dict:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.run_teeth_seg import segment_teeth
        from scripts.segment_soft_tissue import segment as segment_soft
        from scripts.masks_to_stl import nifti_to_stl

        nifti = data_dir / "nifti" / "patient.nii.gz"
        if not nifti.exists():
            raise FileNotFoundError(f"NIfTI not found: {nifti}. Run dcm_to_nifti.py first.")

        seg_dir = data_dir / "seg"
        stl_dir = data_dir / "stl"
        stl_dir.mkdir(exist_ok=True)

        segment_teeth(str(nifti), str(seg_dir / "teeth"))
        segment_soft(str(nifti), str(seg_dir / "soft"))

        masks = list((seg_dir / "teeth").glob("*.nii.gz")) + list((seg_dir / "soft").glob("*.nii.gz"))
        for mask in masks:
            nifti_to_stl(mask, stl_dir / f"{mask.parent.name}_{mask.stem.replace('.nii','')}.stl")

        stls = list(stl_dir.glob("*.stl"))
        return {"stl_dir": str(stl_dir), "count": len(stls)}
