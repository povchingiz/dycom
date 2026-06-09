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


def segment(input_path: str, output_dir: str) -> None:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    print(f"Volume shape: {arr.shape}, HU range: {arr.min():.0f}–{arr.max():.0f}")

    skin = ((arr >= -700) & (arr <= -200)).astype(np.uint8)
    soft = ((arr > -200) & (arr <= 200)).astype(np.uint8)

    for name, mask in [("skin", skin), ("soft_tissue", soft)]:
        out_img = sitk.GetImageFromArray(mask)
        out_img.CopyInformation(img)
        path = out / f"{name}.nii.gz"
        sitk.WriteImage(out_img, str(path))
        voxels = int(mask.sum())
        vol_cc = voxels * (0.4 ** 3) / 1000
        print(f"  {name}: {voxels} voxels ({vol_cc:.1f} cc) → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="data/nifti/patient.nii.gz")
    parser.add_argument("-o", "--output", default="data/seg/soft")
    args = parser.parse_args()
    segment(args.input, args.output)
