import shutil
from pathlib import Path
from pipeline.phases.base import Phase

FEB_TEMPLATE = """\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <!-- Mooney-Rivlin soft tissue, orthognathic surgery scenario -->
  <!-- TODO: populate mesh from Phase 2 tet mesh (data/mesh/patient.vtk) -->
  <!-- Material properties:
       bone:        E=15 GPa, nu=0.3
       muscle:      c1=25kPa, c2=0 (neo-Hookean)
       fat:         c1=1.5kPa, c2=0
       skin:        c1=100kPa, c2=0
       Source: Mollemans et al. 2007, Kim et al. 2010 -->
</febio_spec>
"""


class Phase3Sim(Phase):
    name = "phase3_simulation"

    def is_available(self) -> tuple[bool, str]:
        if shutil.which("febio3") or shutil.which("febio"):
            return True, ""
        return False, "FEBio not installed (https://febio.org/downloads/)"

    def artifacts_exist(self, data_dir: Path) -> bool:
        return (data_dir / "sim" / "deformed.vtk").exists()

    def run(self, state, data_dir: Path) -> dict:
        import subprocess
        feb_bin = shutil.which("febio3") or shutil.which("febio")
        sim_dir = data_dir / "sim"
        sim_dir.mkdir(exist_ok=True)
        feb_file = sim_dir / "model.feb"
        feb_file.write_text(FEB_TEMPLATE)

        result = subprocess.run(
            [feb_bin, "-i", str(feb_file)],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FEBio: {result.stderr[:300]}")
        return {"deformed_vtk": str(sim_dir / "deformed.vtk")}

    def run_fallback(self, state, data_dir: Path) -> dict:
        # Phase 2 already produced before/after meshes — use them as sim output
        p2 = state.phase("phase2_meshing")
        after = p2.get("artifacts", {}).get("after")
        if after and Path(after).exists():
            print("[phase3] FEBio absent — using Phase 2 distance-weighted meshes as simulation output")
            return {"method": "phase2_proxy", "after_dir": after}
        raise NotImplementedError("Phase 2 artifacts not available")
