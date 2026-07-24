# Face Simulation Pipeline (FaceSim)

A fully open-source pipeline that predicts **how a patient's face will look after
dental or jaw surgery** — before the procedure happens.

Given one CBCT scan, the pipeline segments teeth and soft tissue, simulates the
soft-tissue response to a jaw movement, and renders a before/after visualization.
No commercial software required.

---

## What problem does this solve?

Surgeons currently rely on experience and 2D X-rays to estimate post-surgical
appearance. This pipeline gives patients and clinicians a 3D preview of outcomes
for procedures like implant placement, orthognathic surgery (jaw repositioning),
tooth extraction, and prosthetics.

Commercial tools that attempt this (Dolphin Imaging, ProPlan CMF) cost $10,000+/yr,
need manual expert input, and are inaccessible to most clinics. FaceSim is
open-source, automated, and reproducible.

---

## Current status

| Phase | Description | Status |
|---|---|---|
| **0 — Setup** | Unpack DICOM, anonymize, verify scan quality | ✅ Complete |
| **1 — Segmentation** | Extract teeth, jawbones, soft tissue as 3D masks (TotalSegmentator) | ✅ Complete |
| **2 — Meshing** | Open3D mesh cleanup, before/after structure | ✅ Complete |
| **3 — Simulation** | Laplacian soft-tissue displacement propagation | ✅ Complete |
| **4 — Render** | Before/after side-by-side render | ✅ Complete |
| **5 — Validation** | Placeholder (real validation needs a paired post-op scan) | ✅ Complete |
| **6 — ML training** | Train nnU-Net on ToothFairy2 to replace TotalSegmentator | 🔄 In progress |

The end-to-end before/after pipeline (phases 0–5) runs today. Phase 6 is a
segmentation-quality upgrade, not a blocker for the demo.

See [ROADMAP.md](ROADMAP.md) for full detail on each phase, decisions, and open work.

---

## What Phase 1 produced

From one CBCT scan (Planmeca ProMax, 409×409×409 voxels @ 0.4mm):

- **56 anatomical masks**: every visible tooth, upper/lower jawbone, maxillary
  sinuses, nerve canals, pharynx
- **2 soft-tissue masks**: skin layer and full soft-tissue volume
- **58 STL mesh files**: input to the simulation stage

Large files are DVC-tracked / shared via Google Drive, not stored in git
(see [Get the data](#get-the-data)).

---

## Project structure

```
facesim/
├── pipeline/                 # Canonical orchestration — state machine, phases 1–6
│   ├── main.py               # Entry point (make pipeline / make train / make status)
│   └── phases/               # p1_seg … p6_train
├── segmentation/             # Segmentation helpers used by pipeline + server
│   ├── dcm_to_nifti.py       # DICOM → NIfTI volume
│   ├── run_teeth_seg.py      # Teeth/jaw segmentation (TotalSegmentator)
│   ├── segment_soft_tissue.py  # Threshold-based skin/soft tissue
│   └── masks_to_stl.py       # NIfTI masks → STL surface meshes
├── training/                 # Phase 6 nnU-Net training (see training/README.md)
│   ├── scripts/              # 00_setup_env, 02_smoke_test, 04_evaluate
│   └── configs/ · notebooks/
├── server/                   # FastAPI web demo (:8000)
├── data/                     # DVC-tracked / Google Drive (anon, nifti, seg, stl)
├── Makefile · Dockerfile · requirements.txt
├── ROADMAP.md                # Full research plan, decisions, progress
├── DEPLOYMENT.md             # GPU-server deployment / ops
└── README.md                 # This file
```

---

## Setup

Requirements: Python 3.12 (not 3.13/3.14 — TotalSegmentator incompatibility),
~8GB RAM for segmentation, ~4GB disk for model weights (auto-downloaded).

```bash
git clone <repo-url> && cd dycom
make setup          # CPU: venv + deps + .env
# or
make setup-gpu      # GPU server: + PyTorch CUDA + nnU-Net + nnUNet_* paths
```

Then edit `.env` (set `DEMO_PASSWORD`, and `SEGMENTATION_DEVICE=cuda` on GPU).

### Get the data

Large files (DICOM, NIfTI, masks, STLs) are not stored in git. Download the
`data/` folder from Google Drive and place it at the project root:

**[Download data folder](https://drive.google.com/drive/folders/1_ejLHSAqT54ABlOMVeU5gx9P-RgiQFti)**

You should end up with `data/anon/`, `data/nifti/`, `data/seg/`, `data/stl/`.

---

## Running the pipeline

Everything goes through the Makefile from the project root:

```bash
make pipeline       # run the full research pipeline (phases 1–6)
make train          # Phase 6 only — nnU-Net training on GPU
make status         # show current pipeline state
```

`make pipeline` == `python pipeline/main.py`. The pipeline is a checkpointed state
machine: each phase records completion, auto-detects existing artifacts, and
resumes from where it stopped. Individual phases: `python pipeline/main.py --phase N`.

Under the hood, Phase 1 calls the helpers in `segmentation/` (DICOM → NIfTI →
TotalSegmentator teeth/jaw + threshold soft tissue → STL). Phase 3 runs the
Laplacian displacement simulation (tunables in `pipeline/phases/p3_sim.py`:
`SCENARIO_MM`, `JAW_RADIUS_MM`, `N_ITER`).

---

## Web demo

```bash
make run            # starts FastAPI at http://0.0.0.0:8000
make stop           # stops it
```

Upload a DICOM → automatic processing (10–15 min on GPU, 3–4h on CPU) → download
a ZIP of STL files. Password-protected; EN/RU/KZ; sessions auto-clean after 7 days.
See [DEPLOYMENT.md](DEPLOYMENT.md) for full GPU-server setup (RunPod, AWS, etc.).

---

## Key design decisions

**Pure-CBCT path (no external face scan):** the Planmeca ProMax scan carries
enough soft-tissue signal (skin HU range present, ~10M voxels).

**TotalSegmentator over DentalSegmentator:** DentalSegmentator needs the 3D Slicer
GUI and can't run headless; TotalSegmentator (ToothFairy3 weights) gives equivalent
coverage via a Python API. Phase 6 aims to replace it with a purpose-trained nnU-Net.

**Laplacian displacement, not FEA:** FEBio/gmsh were abandoned — the skin STL is
not watertight, so tetrahedral meshing crashed. The pipeline instead propagates
jaw displacement to jaw-adjacent skin nodes via a sparse Laplacian solve
(fully vectorized, ~60s). See ROADMAP "Known Issues" for the trail.

---

## Data privacy

The patient DICOM was anonymized before any processing (`PatientName`,
`PatientID` → `ANON_001`; birth date cleared). Raw identified files are never
committed; the original zip is `.gitignore`d.

---

## Reproducibility

Scripts are deterministic given the same input. Weights are versioned
(TotalSegmentator v3.x, ToothFairy3 weights; training on ToothFairy2 Dataset112).

```bash
docker build -t facesim .
docker run -v $(pwd)/data:/app/data facesim
```

---

## References

- Mollemans et al. (2007) — soft-tissue prediction benchmarks
- Kim et al. — material properties for facial soft-tissue FEA
- TotalSegmentator: Wasserthal et al. (2023), RSNA Radiology AI
- ToothFairy2/3 dataset: MICCAI
- FEBio (evaluated, not used): Maas et al. (2012), J. Biomech. Eng.

---

## Team / contact

- Chingiz — yertaychingiz@gmail.com · tg: @povchingiz
- Sabina — sbsqbiz@gmail.com · tg: @sab_realism
