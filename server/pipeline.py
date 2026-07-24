"""
Pipeline orchestrator for FaceSim segmentation.
Runs the 4 segmentation steps with progress callbacks and error handling.
"""
import os
import pathlib
import shutil
from typing import Callable, Optional

DEVICE = os.getenv("SEGMENTATION_DEVICE", "cpu")

# Import functions from segmentation/
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from segmentation.dcm_to_nifti import convert as convert_dcm_to_nifti
from segmentation.run_teeth_seg import segment_teeth
from segmentation.segment_soft_tissue import segment as segment_soft_tissue
from segmentation.masks_to_stl import nifti_to_stl


class PipelineError(Exception):
    """Custom exception for pipeline failures."""
    def __init__(self, step: str, message: str):
        self.step = step
        self.message = message
        super().__init__(f"Step '{step}' failed: {message}")


def run_pipeline(
    session_id: str,
    dicom_path: str,
    session_dir: pathlib.Path,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> str:
    """
    Run the complete segmentation pipeline.
    
    Args:
        session_id: Unique session identifier
        dicom_path: Path to uploaded DICOM file
        session_dir: Session directory for outputs
        progress_callback: Optional callback function(status_message, percentage)
    
    Returns:
        Path to ZIP file with results
    
    Raises:
        PipelineError: If any step fails
    """
    
    def progress(msg: str, pct: int):
        if progress_callback:
            progress_callback(msg, pct)
    
    # Create output directories
    nifti_dir = session_dir / "nifti"
    seg_teeth_dir = session_dir / "seg" / "teeth"
    seg_soft_dir = session_dir / "seg" / "soft"
    stl_dir = session_dir / "stl"
    
    nifti_dir.mkdir(parents=True, exist_ok=True)
    seg_teeth_dir.mkdir(parents=True, exist_ok=True)
    seg_soft_dir.mkdir(parents=True, exist_ok=True)
    stl_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert DICOM to NIfTI
    progress("Converting DICOM to NIfTI...", 10)
    nifti_path = nifti_dir / "patient.nii.gz"
    try:
        convert_dcm_to_nifti(str(dicom_path), str(nifti_path))
    except Exception as e:
        raise PipelineError("DICOM Conversion", str(e))
    progress("DICOM conversion complete", 25)
    
    # Step 2: Segment teeth and jawbones
    progress("Segmenting teeth and jawbones (this takes ~10 min)...", 30)
    try:
        segment_teeth(str(nifti_path), str(seg_teeth_dir), device=DEVICE)
    except Exception as e:
        raise PipelineError("Teeth Segmentation", str(e))
    progress("Teeth segmentation complete", 50)
    
    # Step 3: Segment soft tissue
    progress("Segmenting soft tissue...", 55)
    try:
        segment_soft_tissue(str(nifti_path), str(seg_soft_dir))
    except Exception as e:
        raise PipelineError("Soft Tissue Segmentation", str(e))
    progress("Soft tissue segmentation complete", 65)
    
    # Step 4: Convert masks to STL meshes
    progress("Generating 3D meshes (STL files)...", 70)
    try:
        masks = list(seg_teeth_dir.glob("*.nii.gz")) + list(seg_soft_dir.glob("*.nii.gz"))
        for mask in masks:
            if mask.parent.name == "teeth":
                out_name = f"teeth_{mask.stem.replace('.nii', '')}.stl"
            else:
                out_name = f"soft_{mask.stem.replace('.nii', '')}.stl"
            out_path = stl_dir / out_name
            nifti_to_stl(mask, out_path)
    except Exception as e:
        raise PipelineError("STL Generation", str(e))
    progress("3D mesh generation complete", 85)
    
    # Step 5: Create ZIP archive
    progress("Preparing download package...", 90)
    zip_path = session_dir / f"results_{session_id}.zip"
    try:
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', stl_dir)
    except Exception as e:
        raise PipelineError("ZIP Creation", str(e))
    progress("Download package ready", 100)
    
    return str(zip_path)
