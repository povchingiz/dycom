"""Convert single-file Planmeca CBCT DICOM to NIfTI."""
import argparse
import pydicom
import numpy as np
import SimpleITK as sitk
from typing import Optional, Callable


def convert(
    dcm_path: str,
    out_path: str,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> None:
    """
    Convert DICOM to NIfTI.
    
    Args:
        dcm_path: Path to input DICOM file
        out_path: Path for output NIfTI file
        progress_callback: Optional callback function(message, percentage)
    """
    if progress_callback:
        progress_callback("Loading DICOM file...", 10)
    
    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array  # (frames, rows, cols)

    if progress_callback:
        progress_callback("Converting to Hounsfield units...", 40)
    
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1))
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0))
    hu = pixels.astype(np.float32) * rescale_slope + rescale_intercept

    # SimpleITK expects (z, y, x) → that's what (frames, rows, cols) already is
    img = sitk.GetImageFromArray(hu)

    if progress_callback:
        progress_callback("Setting voxel spacing...", 70)
    
    voxel_mm = float(getattr(ds, 'SliceThickness', 0.4))
    img.SetSpacing((voxel_mm, voxel_mm, voxel_mm))

    if progress_callback:
        progress_callback("Writing NIfTI file...", 90)
    
    sitk.WriteImage(img, out_path)
    
    if progress_callback:
        progress_callback("Conversion complete", 100)
    
    print(f"Saved NIfTI: {out_path}")
    print(f"  Shape: {img.GetSize()}, Spacing: {img.GetSpacing()}")
    print(f"  HU range: {hu.min():.0f} to {hu.max():.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dcm_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    convert(args.dcm_path, args.out_path)
