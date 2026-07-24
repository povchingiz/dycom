"""
Threshold-based soft tissue segmentation from CBCT.
Skin: -700 to -200 HU  (fat/subcutaneous layer)
Soft tissue: -200 to 200 HU
Output: data/seg/soft/skin.nii.gz, soft_tissue.nii.gz
"""
import argparse
import pathlib
import numpy as np
import SimpleITK as sitk
from typing import Optional, Callable


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

    skin = ((arr >= -700) & (arr <= -200)).astype(np.uint8)
    soft = ((arr > -200) & (arr <= 200)).astype(np.uint8)

    if progress_callback:
        progress_callback("Saving skin mask...", 60)
    
    for name, mask in [("skin", skin), ("soft_tissue", soft)]:
        out_img = sitk.GetImageFromArray(mask)
        out_img.CopyInformation(img)
        path = out / f"{name}.nii.gz"
        sitk.WriteImage(out_img, str(path))
        voxels = int(mask.sum())
        vol_cc = voxels * (0.4 ** 3) / 1000
        print(f"  {name}: {voxels} voxels ({vol_cc:.1f} cc) → {path}")
        
        if progress_callback:
            progress_callback(f"Saved {name} mask", 80)
    
    if progress_callback:
        progress_callback("Soft tissue segmentation complete", 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="data/nifti/patient.nii.gz")
    parser.add_argument("-o", "--output", default="data/seg/soft")
    args = parser.parse_args()
    segment(args.input, args.output)
