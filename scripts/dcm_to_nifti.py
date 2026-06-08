"""Convert single-file Planmeca CBCT DICOM to NIfTI."""
import argparse
import pydicom
import numpy as np
import SimpleITK as sitk


def convert(dcm_path: str, out_path: str) -> None:
    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array  # (frames, rows, cols)

    rescale_slope = float(getattr(ds, 'RescaleSlope', 1))
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0))
    hu = pixels.astype(np.float32) * rescale_slope + rescale_intercept

    # SimpleITK expects (z, y, x) → that's what (frames, rows, cols) already is
    img = sitk.GetImageFromArray(hu)

    voxel_mm = float(getattr(ds, 'SliceThickness', 0.4))
    img.SetSpacing((voxel_mm, voxel_mm, voxel_mm))

    sitk.WriteImage(img, out_path)
    print(f"Saved NIfTI: {out_path}")
    print(f"  Shape: {img.GetSize()}, Spacing: {img.GetSpacing()}")
    print(f"  HU range: {hu.min():.0f} to {hu.max():.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dcm_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    convert(args.dcm_path, args.out_path)
