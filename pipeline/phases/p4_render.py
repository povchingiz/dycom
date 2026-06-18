from pathlib import Path
from pipeline.phases.base import Phase


class Phase4Render(Phase):
    name = "phase4_render"

    def artifacts_exist(self, data_dir: Path) -> bool:
        render_dir = data_dir / "renders"
        return render_dir.exists() and len(list(render_dir.glob("*.png"))) >= 2

    def run(self, state, data_dir: Path) -> dict:
        return self._render(state, data_dir)

    def run_fallback(self, state, data_dir: Path) -> dict:
        return self._render(state, data_dir)

    def _get_dirs(self, state, data_dir: Path) -> tuple[Path, Path]:
        p3 = state.phase("phase3_simulation")
        p2 = state.phase("phase2_meshing")
        after = (
            p3.get("artifacts", {}).get("after_dir")
            or p2.get("artifacts", {}).get("after")
            or str(data_dir / "mesh" / "after")
        )
        before = (
            p2.get("artifacts", {}).get("before")
            or str(data_dir / "mesh" / "before")
        )
        if not Path(before).exists() or not Path(after).exists():
            raise RuntimeError("data/mesh/before or after not found — run Phase 2 first")
        return Path(before), Path(after)

    def _render(self, state, data_dir: Path) -> dict:
        import trimesh

        before_dir, after_dir = self._get_dirs(state, data_dir)
        render_dir = data_dir / "renders"
        render_dir.mkdir(exist_ok=True)
        outputs = []

        for label, mesh_dir in [("before", before_dir), ("after", after_dir)]:
            skin_path = mesh_dir / "skin.stl"
            jaw_path = mesh_dir / "jaw.stl"
            if not skin_path.exists():
                continue

            scene = trimesh.Scene()
            skin = trimesh.load(str(skin_path))
            skin.visual.face_colors = [210, 180, 160, 200]
            scene.add_geometry(skin, node_name="skin")

            if jaw_path.exists():
                jaw = trimesh.load(str(jaw_path))
                jaw.visual.face_colors = [245, 235, 215, 255]
                scene.add_geometry(jaw, node_name="jaw")

            png_path = render_dir / f"{label}.png"
            try:
                png = scene.save_image(resolution=[1024, 768])
                if png:
                    png_path.write_bytes(png)
                    outputs.append(str(png_path))
                    print(f"[phase4] saved {png_path}")
            except Exception as e:
                # Headless: save mesh stats instead
                info = render_dir / f"{label}_info.txt"
                info.write_text(
                    f"skin vertices: {len(skin.vertices)}, faces: {len(skin.faces)}\n"
                    f"render failed (headless?): {e}\n"
                    f"open in MeshLab: {skin_path}"
                )
                outputs.append(str(info))

        return {"renders": outputs, "render_dir": str(render_dir)}
