"""
Phase 3 — FEBio simulation (orthognathic surgery: 5mm mandible advancement).

Flow:
  1. Simplify + tet-mesh skin surface (gmsh)
  2. Load jaw surface as rigid body
  3. Generate .feb XML (Mooney-Rivlin skin, rigid jaw, skull-base fixed)
  4. Run febio3
  5. Parse nodal displacement logfile → deformed mesh
  6. Save deformed.vtk + before/after STLs for Phase 4
"""
from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from pipeline.phases.base import Phase

SCENARIO_MM = 5.0          # mandible advancement along Y (anterior) axis
SKIN_MAX_FACES = 20_000    # simplify before tet-meshing (original ~200k)
MESH_SIZE_MM = 5.0         # gmsh max element size in mm


class Phase3Sim(Phase):
    name = "phase3_simulation"

    def is_available(self) -> tuple[bool, str]:
        if shutil.which("febio3") or shutil.which("febio4") or shutil.which("febio"):
            return True, ""
        return False, "FEBio not in PATH — run: make setup-febio"

    def artifacts_exist(self, data_dir: Path) -> bool:
        return (data_dir / "sim" / "deformed.vtk").exists()

    # ── main entry ────────────────────────────────────────────────────

    def run(self, state, data_dir: Path) -> dict:
        sim_dir = data_dir / "sim"
        sim_dir.mkdir(exist_ok=True)

        jaw_stl = data_dir / "stl" / "teeth_lower_jawbone.stl"
        skin_stl = data_dir / "stl" / "soft_skin.stl"
        for p in (jaw_stl, skin_stl):
            if not p.exists():
                raise FileNotFoundError(f"Required STL not found: {p}")

        print(f"[phase3] meshing skin volume (simplify to {SKIN_MAX_FACES} faces, {MESH_SIZE_MM}mm elements)...")
        nodes, tets = self._mesh_skin(skin_stl, sim_dir)
        print(f"[phase3] skin mesh: {len(nodes)} nodes, {len(tets)} tets")

        jaw_verts, _ = self._load_surface(jaw_stl)
        print(f"[phase3] jaw surface: {len(jaw_verts)} verts for contact proximity")

        feb_path = sim_dir / "model.feb"
        self._generate_feb(nodes, tets, jaw_verts, feb_path, sim_dir)
        print(f"[phase3] wrote {feb_path}")

        print("[phase3] running FEBio (10-30 min)...")
        self._run_febio(feb_path, sim_dir)

        log_path = sim_dir / "displacements.csv"
        deformed_nodes = self._parse_logfile(log_path, nodes)
        max_disp = float(np.linalg.norm(deformed_nodes - nodes, axis=1).max())
        print(f"[phase3] max displacement: {max_disp:.2f} mm")

        deformed_vtk = sim_dir / "deformed.vtk"
        self._save_vtk(deformed_nodes, tets, deformed_vtk)

        before_dir, after_dir = self._save_scene_stls(
            nodes, tets, deformed_nodes, jaw_stl, data_dir
        )

        return {
            "deformed_vtk": str(deformed_vtk),
            "before": str(before_dir),
            "after": str(after_dir),
            "after_dir": str(after_dir),   # Phase 4 reads this key
            "n_nodes": len(nodes),
            "n_tets": len(tets),
            "max_disp_mm": round(max_disp, 3),
            "scenario_mm": SCENARIO_MM,
        }

    def run_fallback(self, state, data_dir: Path) -> dict:
        p2 = state.phase("phase2_meshing")
        after = p2.get("artifacts", {}).get("after")
        if after and Path(after).exists():
            print("[phase3] FEBio absent — using Phase 2 distance-weighted meshes as proxy")
            return {"method": "phase2_proxy", "after_dir": after}
        raise NotImplementedError("Phase 2 artifacts not available for fallback")

    # ── skin meshing ──────────────────────────────────────────────────

    def _mesh_skin(self, skin_stl: Path, work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        import gmsh
        import trimesh

        mesh = trimesh.load(str(skin_stl), force="mesh")
        if len(mesh.faces) > SKIN_MAX_FACES:
            mesh = mesh.simplify_quadric_decimation(SKIN_MAX_FACES)
        trimesh.repair.fix_normals(mesh)
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass  # fill_holes can fail on complex meshes; gmsh will handle minor gaps

        repaired = work_dir / "_skin_repaired.stl"
        mesh.export(str(repaired))

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE_MM)

        gmsh.merge(str(repaired))
        gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)  # permissive angle for skin
        gmsh.model.mesh.createGeometry()
        gmsh.model.mesh.generate(3)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        all_nodes = coords.reshape(-1, 3)
        tag_to_idx = {t: i for i, t in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        tet_list = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 4:  # linear tet4
                arr = enodes.reshape(-1, 4)
                mapped = np.array([[tag_to_idx[t] for t in row] for row in arr], dtype=int)
                tet_list.append(mapped)
        tets = np.vstack(tet_list) if tet_list else None

        gmsh.finalize()
        repaired.unlink(missing_ok=True)

        if tets is None or len(tets) == 0:
            raise RuntimeError(
                "gmsh produced no tet elements — skin STL may not be watertight enough to mesh.\n"
                "Try increasing SKIN_MAX_FACES or check data/stl/soft_skin.stl."
            )

        return all_nodes, tets

    def _load_surface(self, stl: Path) -> tuple[np.ndarray, np.ndarray]:
        import trimesh
        m = trimesh.load(str(stl), force="mesh")
        return np.array(m.vertices), np.array(m.faces)

    # ── FEB XML generation ────────────────────────────────────────────

    def _generate_feb(
        self,
        nodes: np.ndarray,
        tets: np.ndarray,
        jaw_verts: np.ndarray,
        feb_path: Path,
        sim_dir: Path,
    ):
        """
        Prescribe displacement on skin nodes adjacent to jaw (no contact solver needed).
        Skull base is fixed. Jaw-adjacent skin nodes are pushed 5mm forward (+Y).
        Elastic material propagates deformation through the skin volume.
        """
        from scipy.spatial import cKDTree

        # Skull base: top 5% by Z (superior direction)
        z_thresh = np.percentile(nodes[:, 2], 95)
        skull_ids = set(int(i) for i in np.where(nodes[:, 2] >= z_thresh)[0])

        # Jaw contact: skin nodes within 10mm of jaw surface
        jaw_tree = cKDTree(jaw_verts)
        dists, _ = jaw_tree.query(nodes)
        jaw_contact_ids = [i for i in np.where(dists < 10.0)[0].tolist() if i not in skull_ids]

        if not jaw_contact_ids:
            raise RuntimeError(
                "No skin nodes found within 10mm of jaw surface — "
                "coordinate systems may not align. Check STL origins."
            )
        print(f"[phase3] jaw contact nodes: {len(jaw_contact_ids)}")

        root = Element("febio_spec", version="4.0")
        SubElement(root, "Module", type="solid")

        ctrl = SubElement(root, "Control")
        SubElement(ctrl, "analysis").text = "STATIC"
        SubElement(ctrl, "time_steps").text = "10"
        SubElement(ctrl, "step_size").text = "0.1"
        SubElement(ctrl, "max_refs").text = "15"
        SubElement(ctrl, "max_ups").text = "10"

        mats = SubElement(root, "Material")
        skin_m = SubElement(mats, "material", id="1", name="Skin", type="Mooney-Rivlin")
        SubElement(skin_m, "c1").text = "100000"  # 100 kPa
        SubElement(skin_m, "c2").text = "0"
        SubElement(skin_m, "k").text = "1000000"  # 1 MPa bulk

        mesh_el = SubElement(root, "Mesh")

        skin_nodes_el = SubElement(mesh_el, "Nodes", name="SkinNodes")
        for i, (x, y, z) in enumerate(nodes):
            SubElement(skin_nodes_el, "node", id=str(i + 1)).text = f"{x:.6f},{y:.6f},{z:.6f}"

        skin_elems_el = SubElement(mesh_el, "Elements", type="tet4", mat="1", name="SkinElems")
        for i, tet in enumerate(tets):
            SubElement(skin_elems_el, "elem", id=str(i + 1)).text = ",".join(str(n + 1) for n in tet)

        skull_ns = SubElement(mesh_el, "NodeSet", name="SkullBase")
        skull_ns.text = ",".join(str(i + 1) for i in sorted(skull_ids))

        jaw_ns = SubElement(mesh_el, "NodeSet", name="JawContact")
        jaw_ns.text = ",".join(str(i + 1) for i in jaw_contact_ids)

        domains = SubElement(root, "MeshDomains")
        SubElement(domains, "SolidDomain", name="SkinElems", mat="Skin")

        bc_sec = SubElement(root, "Boundary")
        fix = SubElement(bc_sec, "bc", name="FixSkull", type="zero displacement", node_set="SkullBase")
        SubElement(fix, "dofs").text = "x,y,z"

        push = SubElement(bc_sec, "bc", name="PushJaw", type="prescribed displacement", node_set="JawContact")
        SubElement(push, "dof").text = "y"
        SubElement(push, "value", lc="1").text = str(SCENARIO_MM)

        ld = SubElement(root, "LoadData")
        lc = SubElement(ld, "load_controller", id="1", type="loadcurve")
        SubElement(lc, "interpolate").text = "LINEAR"
        pts = SubElement(lc, "points")
        SubElement(pts, "pt").text = "0,0"
        SubElement(pts, "pt").text = "1,1"

        out_el = SubElement(root, "Output")
        lf_el = SubElement(out_el, "logfile", file=str(sim_dir / "displacements.csv"))
        SubElement(lf_el, "node_data", data="ux;uy;uz", delim=",", node_set="SkinNodes")

        xml_body = minidom.parseString(tostring(root)).toprettyxml(indent="  ", encoding=None)
        # FEBio requires ISO-8859-1 encoding declaration
        xml_body = xml_body.replace(
            '<?xml version="1.0" ?>',
            '<?xml version="1.0" encoding="ISO-8859-1"?>',
            1,
        )
        feb_path.write_text(xml_body)

    # ── FEBio execution ───────────────────────────────────────────────

    def _run_febio(self, feb_path: Path, sim_dir: Path):
        feb_bin = shutil.which("febio3") or shutil.which("febio4") or shutil.which("febio")
        result = subprocess.run(
            [feb_bin, "-i", str(feb_path)],
            capture_output=True, text=True, timeout=7200, cwd=str(sim_dir),
        )
        if result.stdout:
            print(result.stdout[-3000:])
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr[-1000:])
            raise RuntimeError(
                f"FEBio exited {result.returncode} — see {sim_dir}/model.log for details"
            )

    # ── output parsing ────────────────────────────────────────────────

    def _parse_logfile(self, log_path: Path, nodes: np.ndarray) -> np.ndarray:
        if not log_path.exists():
            raise FileNotFoundError(
                f"FEBio displacement logfile not found: {log_path}\n"
                "Check data/sim/model.log for FEBio errors."
            )

        displacements = np.zeros_like(nodes)
        current_block: list[str] = []
        blocks: list[list[str]] = []

        with open(log_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("*Step"):
                    if current_block:
                        blocks.append(current_block)
                    current_block = []
                elif stripped and not stripped.startswith("*"):
                    current_block.append(stripped)
        if current_block:
            blocks.append(current_block)

        if not blocks:
            raise RuntimeError(
                "FEBio logfile is empty — simulation may not have converged.\n"
                f"Check {log_path.parent}/model.log"
            )

        for line in blocks[-1]:
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    idx = int(parts[0].strip()) - 1
                    ux, uy, uz = float(parts[1]), float(parts[2]), float(parts[3])
                    if 0 <= idx < len(nodes):
                        displacements[idx] = [ux, uy, uz]
                except (ValueError, IndexError):
                    continue

        return nodes + displacements

    # ── save outputs ──────────────────────────────────────────────────

    def _save_vtk(self, nodes: np.ndarray, tets: np.ndarray, out: Path):
        import meshio
        meshio.write(str(out), meshio.Mesh(points=nodes, cells=[("tetra", tets)]))

    def _surface_from_tets(self, tets: np.ndarray) -> np.ndarray:
        face_count: Counter = Counter()
        for tet in tets:
            for tri in (
                (tet[0], tet[1], tet[2]),
                (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[3]),
                (tet[1], tet[2], tet[3]),
            ):
                face_count[tuple(sorted(tri))] += 1
        return np.array([list(f) for f, c in face_count.items() if c == 1])

    def _save_scene_stls(
        self,
        orig_nodes: np.ndarray,
        tets: np.ndarray,
        deformed_nodes: np.ndarray,
        jaw_stl: Path,
        data_dir: Path,
    ) -> tuple[Path, Path]:
        import trimesh

        faces = self._surface_from_tets(tets)
        before_dir = data_dir / "mesh" / "before"
        after_dir = data_dir / "mesh" / "after"
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)

        trimesh.Trimesh(orig_nodes, faces).export(str(before_dir / "skin.stl"))
        trimesh.Trimesh(deformed_nodes, faces).export(str(after_dir / "skin.stl"))
        shutil.copy(jaw_stl, before_dir / "jaw.stl")
        shutil.copy(jaw_stl, after_dir / "jaw.stl")  # rigid body — geometry unchanged

        return before_dir, after_dir
