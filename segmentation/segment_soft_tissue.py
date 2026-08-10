"""
Threshold-based soft tissue segmentation from CBCT.

skin.nii.gz        the body envelope: everything above the air threshold, closed,
                   hole-filled, largest connected component only. Its outer
                   surface IS the face — that is what Phase 3 deforms.
soft_tissue.nii.gz muscle/glandular band (-200 to 200 HU), cleaned the same way.

Why not the raw HU band: the previous version wrote `(-700 <= HU <= -200)`
straight to disk. That is the fat/air transition shell, so the mesh built from it
came out as ~96k disconnected speckles with 70k non-manifold edges — unusable as
a simulation domain and the reason the skin STL was never watertight.
"""
import argparse
import os
import pathlib
import numpy as np
import SimpleITK as sitk
from typing import Optional, Callable

# Air/tissue boundary. CBCT HU calibration drifts between scanners, so this is
# env-overridable rather than baked in.
AIR_HU = float(os.getenv("FACESIM_AIR_HU", "-500"))
CLOSING_RADIUS = int(os.getenv("FACESIM_CLOSING_RADIUS", "2"))  # voxels


def segment(
    input_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> None:
    """
    Segment soft tissue from NIfTI volume using HU thresholds.
    
    Args:
        input_path: Path to input NIfTI file
        output_dir: Directory for output masks
        progress_callback: Optional callback function(message, percentage)
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Loading volume...", 10)
    
    print(f"Loading {input_path}...")
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    if progress_callback:
        progress_callback("Applying HU thresholds...", 40)
    
    print(f"Volume shape: {arr.shape}, HU range: {arr.min():.0f}–{arr.max():.0f}")

    skin = (arr > AIR_HU).astype(np.uint8)
    soft = ((arr > -200) & (arr <= 200)).astype(np.uint8)

    if progress_callback:
        progress_callback("Cleaning masks...", 60)

    voxel_cc = float(np.prod(img.GetSpacing())) / 1000.0   # mm³ → cc, per scanner

    for name, mask in [("skin", skin), ("soft_tissue", soft)]:
        cleaned = _largest_component(_close_and_fill(mask, img))
        out_img = sitk.GetImageFromArray(cleaned)
        out_img.CopyInformation(img)
        path = out / f"{name}.nii.gz"
        sitk.WriteImage(out_img, str(path))
        voxels = int(cleaned.sum())
        raw_voxels = int(mask.sum())
        print(f"  {name}: {voxels} voxels ({voxels * voxel_cc:.1f} cc) → {path}"
              f"   [raw threshold gave {raw_voxels}, cleanup kept {100 * voxels / max(raw_voxels, 1):.1f}%]")

        if progress_callback:
            progress_callback(f"Saved {name} mask", 80)

    if progress_callback:
        progress_callback("Soft tissue segmentation complete", 100)


def _close_and_fill(mask: np.ndarray, ref: "sitk.Image") -> "sitk.Image":
    """Morphological closing (bridge scanner noise) then hole filling, so the
    envelope is solid instead of a shell full of internal cavities."""
    m = sitk.GetImageFromArray(mask)
    m.CopyInformation(ref)
    m = sitk.BinaryMorphologicalClosing(m, [CLOSING_RADIUS] * 3, sitk.sitkBall)
    return sitk.BinaryFillhole(m)


def _largest_component(img: "sitk.Image") -> np.ndarray:
    """Keep only the biggest connected component — drops the speckle islands that
    thresholding always leaves in CBCT air."""
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(img), sortByObjectSize=True)
    return (sitk.GetArrayFromImage(cc) == 1).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="data/nifti/patient.nii.gz")
    parser.add_argument("-o", "--output", default="data/seg/soft")
    args = parser.parse_args()
    segment(args.input, args.output)
