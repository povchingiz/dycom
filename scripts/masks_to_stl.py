"""
Convert NIfTI mask files to STL meshes.
Input:  data/seg/**/*.nii.gz
Output: data/stl/*.stl (one per structure, decimated + smoothed)
"""
import argparse
import pathlib
import numpy as np
import SimpleITK as sitk
from skimage import measure
from typing import Optional, Callable


def nifti_to_stl(
    mask_path: pathlib.Path,
    out_path: pathlib.Path,
    decimate: float = 0.4,
    smooth_iter: int = 20,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> bool:
    """
    Convert NIfTI mask to STL mesh.
    
    Args:
        mask_path: Path to input NIfTI mask
        out_path: Path for output STL file
        decimate: Decimation factor (0-1)
        smooth_iter: Smoothing iterations
        progress_callback: Optional callback function(message, percentage)
    
    Returns:
        True if successful, False if mask was empty
    """
    if progress_callback:
        progress_callback(f"Loading {mask_path.name}...", 10)
    
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()  # (x, y, z)

    if arr.max() == 0:
        print(f"  SKIP (empty): {mask_path.name}")
        return False

    if progress_callback:
        progress_callback("Running marching cubes...", 40)
    
    verts, faces, normals, _ = measure.marching_cubes(
        arr, level=0.5, spacing=(spacing[2], spacing[1], spacing[0])
    )

    # Decimation: cap at max_faces to keep files manageable
    max_faces = 200_000
    if len(faces) > max_faces:
        if progress_callback:
            progress_callback(f"Decimating mesh ({len(faces)} → {max_faces} faces)...", 60)
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]

    if progress_callback:
        progress_callback("Writing STL file...", 80)
    
    # Write ASCII STL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"solid {out_path.stem}\n")
        for tri in faces:
            v = verts[tri]
            n = np.cross(v[1] - v[0], v[2] - v[0])
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for pt in v:
                f.write(f"      vertex {pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {out_path.stem}\n")

    if progress_callback:
        progress_callback(f"Saved {len(faces)} faces", 100)
    
    print(f"  Saved: {out_path} ({len(faces)} faces)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="data/seg")
    parser.add_argument("-o", "--output_dir", default="data/stl")
    parser.add_argument("--decimate", type=float, default=0.4)
    args = parser.parse_args()

    seg_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    masks = list(seg_dir.rglob("*.nii.gz"))
    print(f"Found {len(masks)} masks in {seg_dir}")

    for mask in masks:
        out = out_dir / f"{mask.parent.name}_{mask.stem.replace('.nii', '')}.stl"
        nifti_to_stl(mask, out, decimate=args.decimate)
