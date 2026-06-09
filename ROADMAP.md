# ROADMAP — Face Sim Pipeline

## Glossary

| Term | Meaning |
|---|---|
| CBCT | Cone-Beam Computed Tomography (3D dental/maxillofacial X-ray) |
| FOV | Field of View (scan volume coverage) |
| MFS | Maxillofacial Surgery (rus: ЧЛХ) |
| FEA | Finite Element Analysis |
| FEBio | Open-source nonlinear FEA solver for biomechanics |
| ICP | Iterative Closest Point (mesh registration algorithm) |
| HU | Hounsfield Units (CT density scale) |
| IRB | Institutional Review Board (ethics approval) |
| DVC | Data Version Control (git-like for large files) |
| STL | Stereolithography mesh format (surface) |
| NIfTI | Neuroimaging Informatics Technology Initiative (volumetric format) |
| VTK | Visualization Toolkit mesh format (volumetric) |
| Tet mesh | Tetrahedral volumetric mesh |
| Mooney-Rivlin | Hyperelastic material model for soft tissue |
| Hausdorff distance | Max surface deviation metric between meshes |
| AMASSS | Automatic Multi-Anatomical Skull Structure Segmentation |
| DentalSegmentator | Open-source tooth/jaw CBCT segmenter |
| TotalSegmentator | Open-source full-body CT segmenter |

## Research Question
Open-source pipeline for predicting aesthetic/functional facial changes (before→after) following dental/MFS procedures. Soft + hard tissue simulation from CBCT (+ optional face scan). No commercial software.

## Hypothesis
Can build personalized "before→after" model (implant, orthognathic surgery, extraction, prosthesis) from CBCT + open tools only, with surface error <2mm.

---

## Stack (all open-source)

| Layer | Tool |
|---|---|
| Bone/teeth segmentation | 3D Slicer + DentalSegmentator + AMASSS-CBCT |
| Soft tissue segmentation | TotalSegmentator / threshold / MediaPipe FaceMesh |
| Face capture (if CBCT lacks soft tissue) | iPhone TrueDepth + Heges → OBJ |
| Registration | Open3D ICP / CloudCompare |
| Meshing | Gmsh / TetGen |
| FEA hard+soft | **FEBio** (hyperelastic, Mooney-Rivlin) |
| Render before/after | Blender + Python API |
| Pipeline orchestration | Python + Snakemake/Nextflow |
| Versioning | git + DVC + Docker |

---

## Phase 0 — Setup (day 1-2)

**Input:** `Baytemirova_S.zip` (raw patient archive, possibly DICOM series)
**Output:** anonymized NIfTI volume + repo skeleton + path decision

- [x] Unpack `Baytemirova_S.zip`
- [x] Verify format (DICOM/NIfTI), FOV (full face?), voxel size
  - Planmeca ProMax CBCT; single DCM (137MB); 409×409×409 @ 0.4mm = 163mm cube; HEAD; 16-bit
- [x] **Critical:** does CBCT capture soft tissue? If no → need external face scan
  - **YES** — skin HU (-500 to -200) = 10M voxels present; soft tissue 30% of volume
- [x] DICOM tag anonymization (PatientName, BirthDate) → `data/anon/patient_anon.dcm`
- [x] Init repo: git + DVC + Docker skeleton
- [x] **Decision gate:** pure-CBCT vs CBCT+face-scan fusion
  - **RESULT: pure-CBCT path** — skin segmentable from scan

## Phase 1 — Segmentation (week 1)

**Input:** anonymized NIfTI from Phase 0
**Output:** labeled mask NIfTI + per-structure STL files (mandible, maxilla, teeth, skin)

**Hard tissue:**
- [x] TotalSegmentator (teeth task, ToothFairy3 weights) → 56/77 non-empty masks
  - lower_jawbone, upper_jawbone, left/right maxillary sinus, IAN canals, crown, pharynx, 32 teeth
  - 21 empty = wisdom teeth + pulps not visible (expected for this FOV/patient)
- [ ] Soft tissue segmentation (threshold-based skin, pure-CBCT path)
- [ ] Merge/review masks → check overlap/gaps
- [ ] Export STL (decimation 0.3-0.5, smoothing)

**Gaps (deferred):**
- Incisive canal — skip
- Missing wisdom teeth — not visible in this scan

**Note:** DentalSegmentator/AMASSS replaced by TotalSegmentator teeth task (ToothFairy3 model, 42 classes). Equivalent coverage.

## Phase 2 — Hard+Soft coupling (week 2)

**Input:** STL meshes from Phase 1
**Output:** tetrahedral VTK mesh with material labels + contact definitions

- [ ] Register bone mesh ↔ soft mesh (ICP, Open3D)
- [ ] Tet mesh for soft tissue (Gmsh/TetGen)
- [ ] Material assignment:
  - bone E ~15 GPa
  - muscle ~50 kPa
  - fat ~3 kPa
  - skin ~200 kPa
  - source: Mollemans, Kim
- [ ] Constraints: bone-muscle attachment, skin sliding contacts

## Phase 3 — Before/after simulation (week 3)

**Input:** tet VTK mesh from Phase 2
**Output:** deformed VTK mesh per scenario + displacement field

- [ ] FEBio setup, nonlinear large deformation
- [ ] Scenarios:
  - Implant → local load → soft response
  - Orthognathic (5mm shift) → soft deformation → new face
  - Extraction → atrophy → tissue collapse
- [ ] Hyperelastic Mooney-Rivlin for soft tissue
- [ ] Export deformed mesh

## Phase 4 — Visualization (week 4)

**Input:** deformed mesh from Phase 3 + patient photo (optional)
**Output:** rendered PNG pairs + heatmap PNG + optional web viewer

- [ ] Import deformed mesh into Blender
- [ ] Texture projection from patient photo
- [ ] Side-by-side before/after render
- [ ] Difference heatmap (Hausdorff distance)
- [ ] Optional: AR preview via three.js webview

## Phase 5 — Validation (week 5)

**Input:** predicted deformed mesh + real post-op scan (paired case)
**Output:** error metrics report (mean/max surface distance)

- [ ] Find 1-2 paired before/after cases
- [ ] Surface distance prediction vs reality
- [ ] Target: <2mm mean error (literature standard)

## Phase 6 — ML Acceleration (future, after Phase 5 validated)

**Goal:** Replace slow FEBio simulation with fast learned model. Physics pipeline becomes training data generator.

**Realistic path:**

1. Download **ToothFairy2** (480 scans, 42 classes) → fine-tune TotalSegmentator on dental CBCT specifically
2. Run improved segmenter on 100+ cases → build FEA meshes at scale
3. Simulate procedures with parameter sweeps → synthetic before/after pairs (e.g. 10 patients × 50 param variations = 500 pairs)
4. Train deformation prediction model (nnUNet or lightweight CNN) on synthetic pairs
5. Validate ML model against HaN-Seg soft tissue ground truth (not just against simulator — avoids circular validation)

**Key datasets:**

| Dataset | Scans | Labels | Purpose |
|---|---|---|---|
| ToothFairy2 | 480 | 42 classes: teeth, mandible, maxilla, sinuses | Main segmentation training |
| HaN-Seg | 42 | Mandible + soft tissue OARs + paired MR | Soft tissue model + validation |
| PDDCA (TCIA) | 40 | Mandible (public domain) | Mandible fine-tune |
| CTooth+ | 168 (22 labeled) | Tooth instances | Semi-supervised teeth |

**Critical gap:** none have paired before/after scans → physics pipeline fills this gap by generating synthetic pairs.

**Sim-to-real risk:** FEBio uses assumed material props (±40% person variance). ML learns simulator biases. Mitigate: final validation must use real post-op scans, not synthetic.

---

## Removed from old plan

- Stress / Failure Index (not needed for aesthetics)
- Mesh convergence study
- HU→density calibration (simplified to material classes)
- Sensitivity analysis ±20%

## Added (critical)

- Soft tissue FEA stack (FEBio + hyperelastic)
- Face acquisition fallback (iPhone scan)
- ICP bone↔face registration
- Blender renderer for final output
- Validation on real post-op cases

---

## Risks / Blockers

1. **CBCT FOV** may not cover full face → external scan needed. Verify in Phase 0.
2. **Soft tissue properties** variable → realistic accuracy ±2-3mm, not surgical-grade.
3. **Validation:** paired before/after scans required. **Open question.**
4. **DentalSegmentator** trained on specific scanners — may fail. Plan B in Phase 1.

## Ethics

- DICOM anonymization before processing
- IRB approval if publishing
- Patient consent

## Reproducibility

- Pin versions: Slicer, DentalSegmentator weights, FEBio, Blender
- Docker image for full pipeline
- DVC for large artifacts (STL, mesh)

---

## Next step

Phase 0 step 1: unpack zip, verify format and FOV.
