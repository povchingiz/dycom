# Face Simulation Pipeline (FaceSim)

A fully open-source pipeline that predicts **how a patient's face will look after dental or jaw surgery** — before the procedure happens.

Given one CBCT scan, the pipeline segments teeth and soft tissue, builds a physics simulation of the face, and renders a before/after visualization. No commercial software required.

---

## What problem does this solve?

Surgeons currently rely on experience and 2D X-rays to estimate post-surgical appearance. This pipeline gives patients and clinicians a 3D preview of outcomes for procedures like:

- Implant placement
- Orthognathic surgery (jaw repositioning)
- Tooth extraction
- Prosthetics

The target accuracy is **<2mm surface error** — the clinical standard for soft tissue prediction.

---

## Current status

| Phase | Description | Status |
|---|---|---|
| **0 — Setup** | Unpack DICOM, anonymize, verify scan quality | ✅ Complete |
| **1 — Segmentation** | Extract teeth, jawbones, soft tissue as 3D masks | ✅ Complete |
| **2 — Mesh coupling** | Register bone and soft tissue, build FEA mesh | 🔲 Next |
| **3 — Simulation** | FEBio physics simulation of surgical scenarios | 🔲 Pending |
| **4 — Visualization** | Blender before/after render, heatmap | 🔲 Pending |
| **5 — Validation** | Compare prediction vs real post-op scan | 🔲 Pending |
| **6 — ML acceleration** | Train fast learned model on physics-generated pairs | 🔲 Future |

See [ROADMAP.md](ROADMAP.md) for full detail on each phase, decisions made, and what is left to do.

---

## What Phase 1 produced

From one CBCT scan (Planmeca ProMax, 409×409×409 voxels @ 0.4mm):

- **56 anatomical masks** (NIfTI format): every visible tooth, upper/lower jawbone, maxillary sinuses, inferior alveolar nerve canals, pharynx
- **2 soft tissue masks**: skin layer and full soft tissue volume
- **58 STL mesh files**: ready for Phase 2 FEA coupling

All large files are shared via Google Drive (not stored in git). See [Get the data](#get-the-data) section below.

---

## Project structure

```
facesim/
├── scripts/                  # All processing scripts
│   ├── dcm_to_nifti.py       # Convert DICOM → NIfTI volume
│   ├── run_teeth_seg.py      # Teeth/jaw segmentation (TotalSegmentator)
│   ├── segment.py            # General segmentation runner
│   ├── segment_soft_tissue.py  # Threshold-based skin/soft tissue
│   └── masks_to_stl.py       # NIfTI masks → STL surface meshes
├── pipeline/                 # Pipeline orchestration (in progress)
├── data/                     # All data — download from Google Drive (see below)
│   ├── anon/                 # Anonymized DICOM
│   ├── nifti/                # NIfTI volume
│   ├── seg/                  # Segmentation masks
│   │   ├── teeth/            # 77 tooth/jaw masks
│   │   └── soft/             # skin + soft_tissue masks
│   └── stl/                  # STL meshes
├── Dockerfile                # Reproducible environment
├── requirements.txt          # Python dependencies
├── ROADMAP.md                # Full research plan, decisions, progress
└── README.md                 # This file
```

---

## Setup

### Requirements

- Python 3.12 (not 3.13/3.14 — TotalSegmentator incompatibility)
- ~8GB free RAM for segmentation
- ~4GB disk for model weights (auto-downloaded on first run)

### Install

```bash
git clone <repo-url>
cd facesim

python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

### Get the data

Large files (DICOM, NIfTI, masks, STLs) are not stored in git. Download the `data/` folder from Google Drive and place it in the project root:

**[Download data folder](https://drive.google.com/drive/folders/1_ejLHSAqT54ABlOMVeU5gx9P-RgiQFti)**

After downloading, your directory should have `data/anon/`, `data/nifti/`, `data/seg/`, `data/stl/` at the project root.

---

## Running the pipeline

Scripts are run in order, one phase at a time. **Run from the project root.**

### Phase 0 — Convert DICOM to NIfTI

```bash
python scripts/dcm_to_nifti.py data/anon/patient_anon.dcm data/nifti/patient.nii.gz
```

### Phase 1a — Teeth and jaw segmentation

This takes **1–3 hours on CPU**. Close other applications first.

```bash
python scripts/run_teeth_seg.py
```

Output: `data/seg/teeth/` — 77 NIfTI masks

### Phase 1b — Soft tissue segmentation

Fast (~30 seconds):

```bash
python scripts/segment_soft_tissue.py
```

Output: `data/seg/soft/skin.nii.gz`, `data/seg/soft/soft_tissue.nii.gz`

### Phase 1c — Export STL meshes

```bash
python scripts/masks_to_stl.py
```

Output: `data/stl/` — 58 STL files

---

## Key design decisions

**Pure-CBCT path chosen (no external face scan):** The Planmeca ProMax scan includes sufficient soft tissue signal (skin HU range present, 10M voxels). iPhone face scan fallback is documented in ROADMAP but not needed for this case.

**TotalSegmentator instead of DentalSegmentator:** DentalSegmentator requires 3D Slicer GUI and cannot run headless. TotalSegmentator with ToothFairy3 weights provides equivalent coverage (42 classes) via Python API.

**Physics-first, ML-second:** The pipeline uses finite element analysis (FEBio) to simulate tissue deformation. Once validated, the physics pipeline will generate synthetic before/after pairs to train a fast ML model (Phase 6).

---

## Data privacy

The patient DICOM was anonymized before any processing:
- `PatientName` → `ANON_001`
- `PatientBirthDate` → empty
- `PatientID` → `ANON_001`

Raw identified files are never committed to git. The original zip is excluded via `.gitignore`.

---

## Reproducibility

All scripts are deterministic given the same input. Model weights are versioned (TotalSegmentator v3.x, ToothFairy3 Dataset113). To reproduce exactly:

```bash
docker build -t facesim .
docker run -v $(pwd)/data:/app/data facesim
```

---

## What comes next (Phase 2)

The next contributor needs to:

1. Register the bone STL meshes against the soft tissue STL using ICP (Open3D)
2. Build a tetrahedral volumetric mesh from the soft tissue surface (Gmsh or TetGen)
3. Assign material properties (bone 15 GPa, muscle 50 kPa, fat 3 kPa, skin 200 kPa)
4. Define bone-muscle attachment constraints and skin sliding contacts

See [ROADMAP.md — Phase 2](ROADMAP.md#phase-2--hardsoft-coupling-week-2) for full task list.

---

## References

- Mollemans et al. (2007) — soft tissue prediction benchmarks
- Kim et al. — material properties for facial soft tissue FEA
- TotalSegmentator: Wasserthal et al. (2023), RSNA Radiology AI
- ToothFairy3 dataset: MICCAI 2025
- FEBio: Maas et al. (2012), Journal of Biomechanical Engineering
