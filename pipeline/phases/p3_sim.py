"""
Phase 3 — Skin displacement simulation (orthognathic surgery: 5mm mandible advancement).

Approach: Laplacian displacement propagation on the skin surface mesh.
  - No gmsh / FEBio required (skin STL from TotalSegmentator is not watertight)
  - Jaw-adjacent skin nodes get 5mm +Y push
  - Skull-base nodes are fixed (0 displacement)
  - Iterative Laplacian smoothing propagates displacement to all other nodes
  - Fast: completes in seconds on any machine
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from pipeline.phases.base import Phase

SCENARIO_MM = 5.0      # mandible advancement along Y (anterior) axis
JAW_RADIUS_MM = 15.0   # skin nodes within this distance get full push
SKULL_PERCENTILE = 95  # top N% by Z are fixed (skull base)
N_ITER = 300           # Laplacian smoothing iterations


class Phase3Sim(Phase):
    name = "phase3_simulation"

    def is_available(self) -> tuple[bool, str]:
        return True, ""  # no external tools required

    def artifacts_exist(self, data_dir: Path) -> bool:
        return (data_dir / "sim" / "deformed.vtk").exists()

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state, data_dir: Path) -> dict:
        import trimesh

        sim_dir = data_dir / "sim"
        sim_dir.mkdir(exist_ok=True)

        jaw_stl = data_dir / "stl" / "teeth_lower_jawbone.stl"
        skin_stl = data_dir / "stl" / "soft_skin.stl"
        for p in (jaw_stl, skin_stl):
            if not p.exists():
                raise FileNotFoundError(f"Required STL not found: {p}")

        skin = trimesh.load(str(skin_stl), force="mesh")
        verts = np.array(skin.vertices, dtype=float)
        faces = np.array(skin.faces, dtype=int)
        print(f"[phase3] skin mesh: {len(verts)} verts, {len(faces)} faces")

        jaw = trimesh.load(str(jaw_stl), force="mesh")
        jaw_verts = np.array(jaw.vertices, dtype=float)

        skull_ids, jaw_ids, free_ids = self._classify_nodes(verts, jaw_verts)
        print(f"[phase3] skull fixed: {len(skull_ids)}  jaw push: {len(jaw_ids)}  free: {len(free_ids)}")

        if not jaw_ids:
            raise RuntimeError(
                f"No skin nodes within {JAW_RADIUS_MM}mm of jaw — STL coordinate systems may not align."
            )

        adjacency = self._build_adjacency(faces, len(verts))

        print(f"[phase3] running Laplacian smoothing ({N_ITER} iterations)...")
        deformed_verts = self._propagate(verts, adjacency, skull_ids, jaw_ids, free_ids)

        max_disp = float(np.linalg.norm(deformed_verts - verts, axis=1).max())
        print(f"[phase3] max displacement: {max_disp:.2f} mm")

        deformed_vtk = sim_dir / "deformed.vtk"
        self._save_vtk(deformed_verts, faces, deformed_vtk)

        before_dir, after_dir = self._save_scene_stls(
            verts, faces, deformed_verts, jaw_stl, data_dir
        )

        return {
            "deformed_vtk": str(deformed_vtk),
            "before": str(before_dir),
            "after": str(after_dir),
            "after_dir": str(after_dir),
            "n_verts": len(verts),
            "max_disp_mm": round(max_disp, 3),
            "scenario_mm": SCENARIO_MM,
            "method": "laplacian",
        }

    def run_fallback(self, state, data_dir: Path) -> dict:
        p2 = state.phase("phase2_meshing")
        after = p2.get("artifacts", {}).get("after")
        if after and Path(after).exists():
            print("[phase3] using Phase 2 proxy meshes as fallback")
            return {"method": "phase2_proxy", "after_dir": after}
        raise NotImplementedError("Phase 2 artifacts not available for fallback")

    # ── node classification ───────────────────────────────────────────

    def _classify_nodes(
        self, verts: np.ndarray, jaw_verts: np.ndarray
    ) -> tuple[set, list, list]:
        from scipy.spatial import cKDTree

        z_thresh = np.percentile(verts[:, 2], SKULL_PERCENTILE)
        skull_ids = set(int(i) for i in np.where(verts[:, 2] >= z_thresh)[0])

        jaw_tree = cKDTree(jaw_verts)
        dists, _ = jaw_tree.query(verts)
        jaw_ids = [i for i in np.where(dists < JAW_RADIUS_MM)[0].tolist() if i not in skull_ids]

        constrained = skull_ids | set(jaw_ids)
        free_ids = [i for i in range(len(verts)) if i not in constrained]

        return skull_ids, jaw_ids, free_ids

    # ── adjacency ─────────────────────────────────────────────────────

    def _build_adjacency(self, faces: np.ndarray, n: int) -> list[list[int]]:
        adj: list[set] = [set() for _ in range(n)]
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            adj[a].update((b, c))
            adj[b].update((a, c))
            adj[c].update((a, b))
        return [list(s) for s in adj]

    # ── Laplacian propagation ─────────────────────────────────────────

    def _propagate(
        self,
        verts: np.ndarray,
        adjacency: list[list[int]],
        skull_ids: set,
        jaw_ids: list,
        free_ids: list,
    ) -> np.ndarray:
        from scipy.sparse import coo_matrix

        n = len(verts)
        # Build sparse averaging matrix W (free nodes only — constrained rows stay zero)
        rows, cols, data = [], [], []
        for i in free_ids:
            nb = adjacency[i]
            if nb:
                w = 1.0 / len(nb)
                for j in nb:
                    rows.append(i)
                    cols.append(j)
                    data.append(w)
        W = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        disp = np.zeros_like(verts)
        jaw_arr = np.array(jaw_ids, dtype=int)
        skull_arr = np.array(list(skull_ids), dtype=int)
        disp[jaw_arr, 1] = SCENARIO_MM

        for iteration in range(N_ITER):
            disp = W @ disp          # sparse multiply — vectorized, fast
            disp[jaw_arr, 1] = SCENARIO_MM   # re-pin jaw nodes
            disp[skull_arr] = 0.0            # re-pin skull nodes
            if iteration % 50 == 49:
                print(f"[phase3]   iter {iteration+1}/{N_ITER}")

        return verts + disp

    # ── save outputs ──────────────────────────────────────────────────

    def _save_vtk(self, verts: np.ndarray, faces: np.ndarray, out: Path):
        import meshio
        meshio.write(str(out), meshio.Mesh(points=verts, cells=[("triangle", faces)]))

    def _save_scene_stls(
        self,
        orig_verts: np.ndarray,
        faces: np.ndarray,
        deformed_verts: np.ndarray,
        jaw_stl: Path,
        data_dir: Path,
    ) -> tuple[Path, Path]:
        import trimesh

        before_dir = data_dir / "mesh" / "before"
        after_dir = data_dir / "mesh" / "after"
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)

        trimesh.Trimesh(orig_verts, faces).export(str(before_dir / "skin.stl"))
        trimesh.Trimesh(deformed_verts, faces).export(str(after_dir / "skin.stl"))
        shutil.copy(jaw_stl, before_dir / "jaw.stl")
        shutil.copy(jaw_stl, after_dir / "jaw.stl")

        return before_dir, after_dir
