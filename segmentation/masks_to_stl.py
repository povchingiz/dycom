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

    # Marching cubes leaves a voxel-staircase surface. Taubin smoothing removes it
    # without the volume shrinkage plain Laplacian smoothing causes — the mesh is a
    # measurement, so shrinking it would bias every downstream distance.
    if smooth_iter > 0:
        if progress_callback:
            progress_callback(f"Smoothing ({smooth_iter} iterations)...", 55)
        verts, faces = _smooth(verts, faces, smooth_iter)

    # Decimation: cap at max_faces to keep files manageable.
    # NB: this used to be `np.random.choice(faces)` — dropping random triangles does
    # not decimate, it punches holes. That is what made every skin STL non-watertight
    # and broke the gmsh/FEBio path. Quadric decimation collapses edges instead, so
    # the surface stays closed.
    max_faces = 200_000
    target_faces = int(min(max_faces, len(faces) * (1.0 - max(0.0, min(decimate, 0.95)))))
    if target_faces < len(faces):
        if progress_callback:
            progress_callback(f"Decimating mesh ({len(faces)} → {target_faces} faces)...", 70)
        verts, faces = _decimate(verts, faces, target_faces)

    if progress_callback:
        progress_callback("Writing STL file...", 85)

    # Binary STL — same geometry as the old ASCII output at ~1/5 the size.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(out_path))

    if progress_callback:
        progress_callback(f"Saved {len(faces)} faces", 100)

    print(f"  Saved: {out_path} ({len(faces)} faces, watertight={mesh.is_watertight})")
    return True


def _smooth(verts: np.ndarray, faces: np.ndarray, iterations: int):
    """Taubin (lambda/mu) smoothing — shrink-free alternative to Laplacian."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=iterations)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _decimate(verts: np.ndarray, faces: np.ndarray, target_faces: int):
    """Quadric edge-collapse decimation. Preserves manifoldness; random face
    dropping does not."""
    target_reduction = 1.0 - (target_faces / len(faces))
    try:
        import fast_simplification
        v, f = fast_simplification.simplify(
            np.ascontiguousarray(verts, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.int32),
            target_reduction,
        )
        return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    except ImportError:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
        ).simplify_quadric_decimation(int(target_faces))
        return np.asarray(m.vertices), np.asarray(m.triangles)


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
