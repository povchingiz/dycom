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

- [ ] Unpack `Baytemirova_S.zip`
- [ ] Verify format (DICOM/NIfTI), FOV (full face?), voxel size
- [ ] **Critical:** does CBCT capture soft tissue? If no → need external face scan
- [ ] DICOM tag anonymization (PatientName, BirthDate)
- [ ] Init repo: git + DVC + Docker skeleton
- [ ] **Decision gate:** pure-CBCT vs CBCT+face-scan fusion
  - **Criteria:** if FOV crops soft tissue above lip OR skin HU absent (-700 to -200 range missing) → fusion path
  - **Criteria:** if full face visible and skin segmentable → pure-CBCT path

## Phase 1 — Segmentation (week 1)

**Input:** anonymized NIfTI from Phase 0
**Output:** labeled mask NIfTI + per-structure STL files (mandible, maxilla, teeth, skin)

**Hard tissue:**
- [ ] DentalSegmentator → mandible, teeth, mandibular canal
- [ ] AMASSS-CBCT → maxilla, sinuses, temporal bone
- [ ] Merge masks → single NIfTI, check overlap/gaps
- [ ] Export STL (decimation 0.3-0.5, smoothing)

**Soft tissue:**
- [ ] TotalSegmentator head module OR
- [ ] Threshold-based skin (-200 to -700 HU) OR
- [ ] MediaPipe FaceMesh with photo (fusion path only)

**Gaps (defer):**
- Incisive canal — manual annotation or skip
- Tooth pulp — skip or threshold inside teeth

**Plan B (DentalSegmentator failure):** fine-tune on 5-10 manual annotations using Slicer

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
