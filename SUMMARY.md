# FaceSim — One-Page Summary

## The Problem

Surgeons performing jaw and dental procedures — implants, jaw repositioning, extractions — cannot show patients what their face will look like after surgery. They rely on 2D X-rays and experience. Patients consent without seeing the outcome.

Commercial software that attempts this (Dolphin Imaging, ProPlan CMF) costs $10,000+/year, requires manual expert input, and is inaccessible to most clinics globally.

---

## What We Are Building

An open-source pipeline that takes one patient CT scan and produces a 3D simulation of facial appearance before and after a surgical procedure — fully automated, no commercial software.

**Input:** One CBCT scan (standard dental 3D X-ray, widely available)
**Output:** Side-by-side 3D render of the patient's face before and after the procedure

---

## How It Works — 6 Phases

| Phase | What happens | Status |
|---|---|---|
| **1. Setup** | Load scan, anonymize patient data, verify quality | ✅ Done |
| **2. Segmentation** | AI identifies every tooth, jawbone, and soft tissue layer as a separate 3D object | ✅ Done |
| **3. Mesh coupling** | Bone and soft tissue objects are connected into one unified model | 🔲 Next |
| **4. Simulation** | Physics engine deforms the face based on the surgical change (e.g. jaw moved 5mm) | 🔲 In progress |
| **5. Visualization** | Before/after 3D render with error heatmap | 🔲 Planned |
| **6. Validation** | Prediction compared against real post-op scan. Target: <2mm error | 🔲 Planned |

After validation: train a fast AI model on the physics outputs — reducing simulation time from hours to seconds.

---

## What Phase 2 (Segmentation) Produced

From one patient scan, the pipeline automatically extracted:

- **32 individual teeth** as separate 3D objects
- **Upper and lower jawbones**
- **Maxillary sinuses and nerve canals**
- **Full skin and soft tissue surface**

58 mesh files total. Each structure is a clean 3D object ready for physics simulation.

---

## Why Open-Source

- No licensing fees — deployable in any clinic or research lab
- Reproducible — every step is versioned, documented, and Dockerized
- Extensible — once validated on one patient, scale to 480+ scans (ToothFairy2 public dataset) to train a fast AI model

---

## Stack

All tools are free and open-source: Python, TotalSegmentator, FEBio, Blender, Open3D.

---

## Code & Data

- GitHub: github.com/povchingiz/dycom (private — request access)
- Data: Google Drive (anonymized patient data, 3D meshes)

---

## Team / Contact

Chingiz — yertaychingiz@gmail.com / tg: @povchingiz
Sabina - sbsqbiz@gmail.com        / tg: @sab_realism
