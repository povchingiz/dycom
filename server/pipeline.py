"""
Pipeline orchestrator for FaceSim segmentation.
Runs the 4 segmentation steps with progress callbacks and error handling.
"""
import json
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
    scenario: Optional[dict] = None,
) -> dict:
    """
    Run the complete segmentation pipeline.
    
    Args:
        session_id: Unique session identifier
        dicom_path: Path to uploaded DICOM file
        session_dir: Session directory for outputs
        progress_callback: Optional callback function(status_message, percentage)
        scenario: Surgical plan (advance_mm / vertical_mm / lateral_mm / pitch_deg).
                  None runs segmentation only.

    Returns:
        {"zip_path": ..., "simulation": {...} | None}

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
    progress("3D mesh generation complete", 80)

    # Step 5: Simulate the surgery. This is the product — segmentation alone just
    # returns meshes of the scan the user already had.
    simulation = None
    if scenario is not None:
        progress("Simulating soft-tissue response...", 82)
        try:
            simulation = _simulate(session_dir, scenario)
        except Exception as e:
            raise PipelineError("Simulation", str(e))
        progress(f"Simulation complete (max {simulation['max_disp_mm']} mm)", 88)

    # Step 6: Create ZIP archive
    progress("Preparing download package...", 92)
    zip_path = session_dir / f"results_{session_id}.zip"
    try:
        # Pack the whole result set, not just stl/, so before/after meshes and the
        # simulation summary travel with it.
        package = session_dir / "package"
        if package.exists():
            shutil.rmtree(package)
        package.mkdir()
        shutil.copytree(stl_dir, package / "stl")
        for extra in ("mesh", "sim"):
            src = session_dir / extra
            if src.exists():
                shutil.copytree(src, package / extra)
        if simulation is not None:
            (package / "simulation.json").write_text(json.dumps(simulation, indent=2))
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', package)
        shutil.rmtree(package, ignore_errors=True)
    except Exception as e:
        raise PipelineError("ZIP Creation", str(e))
    progress("Download package ready", 100)

    return {"zip_path": str(zip_path), "simulation": simulation}


def _simulate(session_dir: pathlib.Path, scenario: dict) -> dict:
    """Run Phase 3 on this session's meshes with the requested surgical plan."""
    from pipeline.phases.p3_sim import Phase3Sim, Scenario

    plan = Scenario(
        advance_mm=float(scenario.get("advance_mm", 5.0)),
        vertical_mm=float(scenario.get("vertical_mm", 0.0)),
        lateral_mm=float(scenario.get("lateral_mm", 0.0)),
        pitch_deg=float(scenario.get("pitch_deg", 0.0)),
    )
    result = Phase3Sim(plan).run(None, session_dir)
    return {
        "scenario": result["scenario"],
        "max_disp_mm": result["max_disp_mm"],
        "mean_disp_mm": result["mean_disp_mm"],
        "frame": result["frame"],
        "method": result["method"],
    }
