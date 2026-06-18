from pathlib import Path
import numpy as np
from pipeline.phases.base import Phase


class Phase2Mesh(Phase):
    name = "phase2_meshing"

    def is_available(self) -> tuple[bool, str]:
        try:
            import gmsh  # noqa
            return True, ""
        except ImportError:
            return False, "gmsh not installed — using distance-weighted fallback"

    def artifacts_exist(self, data_dir: Path) -> bool:
        after_dir = data_dir / "mesh" / "after"
        return after_dir.exists() and len(list(after_dir.glob("*.stl"))) >= 2

    def run(self, state, data_dir: Path) -> dict:
        import gmsh
        stl_dir = data_dir / "stl"
        mesh_dir = data_dir / "mesh"
        mesh_dir.mkdir(exist_ok=True)

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        for stl in stl_dir.glob("*.stl"):
            gmsh.merge(str(stl))
        gmsh.model.mesh.classifySurfaces(3.14159, True, True, 3.14159)
        gmsh.model.mesh.createGeometry()
        gmsh.model.mesh.generate(3)
        out = str(mesh_dir / "patient.vtk")
        gmsh.write(out)
        gmsh.finalize()
        return {"vtk": out}

    def run_fallback(self, state, data_dir: Path) -> dict:
        """Rigid jaw displacement + distance-weighted skin deformation (no FEA required)."""
        import trimesh
        from scipy.spatial import cKDTree

        stl_dir = data_dir / "stl"
        jaw_stl = stl_dir / "teeth_lower_jawbone.stl"
        skin_stl = stl_dir / "soft_skin.stl"

        if not jaw_stl.exists():
            raise FileNotFoundError(f"Expected: {jaw_stl}")
        if not skin_stl.exists():
            raise FileNotFoundError(f"Expected: {skin_stl}")

        jaw = trimesh.load(str(jaw_stl))
        skin = trimesh.load(str(skin_stl))

        jaw_orig_verts = jaw.vertices.copy()

        # Surgery scenario: move lower jaw 5 mm forward (+Y in CBCT space)
        translation = np.array([0.0, 5.0, 0.0])
        jaw_after = jaw.copy()
        jaw_after.apply_translation(translation)

        # Deform skin: each vertex moves proportionally to proximity to jaw
        tree = cKDTree(jaw_orig_verts)
        dists, _ = tree.query(skin.vertices, workers=-1)
        sigma = 20.0  # mm influence radius
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
        skin_after = skin.copy()
        skin_after.vertices = skin.vertices + weights[:, None] * translation[None, :]

        before_dir = data_dir / "mesh" / "before"
        after_dir = data_dir / "mesh" / "after"
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)

        jaw.export(str(before_dir / "jaw.stl"))
        skin.export(str(before_dir / "skin.stl"))
        jaw_after.export(str(after_dir / "jaw.stl"))
        skin_after.export(str(after_dir / "skin.stl"))

        print(f"[phase2] jaw moved {translation} mm, skin deformed via distance weighting (σ={sigma}mm)")
        return {
            "method": "distance_weighted",
            "translation_mm": translation.tolist(),
            "before": str(before_dir),
            "after": str(after_dir),
        }
