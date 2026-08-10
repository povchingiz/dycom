"""
Anatomical frame detection for the patient's mesh space.

Why this exists: the STLs are built by marching cubes over the raw NIfTI array
(see segmentation/masks_to_stl.py), so their axis order is whatever the DICOM →
NIfTI conversion produced — NOT a fixed "X=left, Y=anterior, Z=superior".
For the current Planmeca patient, axis 0 is inferior-positive, axis 1 is
posterior-positive and axis 2 is left-positive; another scanner or another export
will differ. Any phase that hardcodes an axis index silently simulates the wrong
surgery, so the frame is derived from the segmentation itself.

Derivation (all from structures Phase 1 already produces):
  superior  = maxilla centroid − mandible centroid      (upper jaw sits above lower)
  lateral   = left structure − right structure          (sinuses, else molar pair)
  anterior  = lateral × superior, sign-checked so incisors end up in front

Lateral is measured before anterior on purpose: a single-side incisor→molar vector
carries a large left-right component (both teeth are on the same side of the arch),
so using it as the primary anterior cue tilts the whole frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# STL basenames written by masks_to_stl.py (`{maskdir}_{maskname}.stl`)
MANDIBLE = "teeth_lower_jawbone"
MAXILLA = "teeth_upper_jawbone"
PHARYNX = "teeth_pharynx"
SINUS_L = "teeth_left_maxillary_sinus"
SINUS_R = "teeth_right_maxillary_sinus"
MOLAR_L = "teeth_lower_left_first_molar_fdi36"
MOLAR_R = "teeth_lower_right_first_molar_fdi46"
INCISOR_L = "teeth_lower_left_central_incisor_fdi31"
INCISOR_R = "teeth_lower_right_central_incisor_fdi41"


@dataclass(frozen=True)
class AnatomicalFrame:
    """Orthonormal patient frame in mesh coordinates (mm)."""

    superior: np.ndarray   # unit vector toward the skull vertex
    anterior: np.ndarray   # unit vector toward the face
    lateral: np.ndarray    # unit vector toward the patient's left
    source: str            # how it was derived — printed into pipeline state

    def to_dict(self) -> dict:
        return {
            "superior": [round(float(v), 4) for v in self.superior],
            "anterior": [round(float(v), 4) for v in self.anterior],
            "lateral": [round(float(v), 4) for v in self.lateral],
            "source": self.source,
        }

    def project(self, vec_mm: dict[str, float]) -> np.ndarray:
        """Anatomical displacement → mesh-space vector.

        vec_mm keys: 'anterior', 'superior', 'lateral' (mm, may be negative)."""
        return (
            float(vec_mm.get("anterior", 0.0)) * self.anterior
            + float(vec_mm.get("superior", 0.0)) * self.superior
            + float(vec_mm.get("lateral", 0.0)) * self.lateral
        )


def _centroid(stl_dir: Path, name: str) -> np.ndarray | None:
    path = stl_dir / f"{name}.stl"
    if not path.exists():
        return None
    import trimesh
    mesh = trimesh.load(str(path), force="mesh")
    return np.asarray(mesh.vertices, dtype=float).mean(axis=0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("degenerate axis vector")
    return v / n


def detect_frame(stl_dir: Path) -> AnatomicalFrame:
    """Derive the patient frame from Phase 1 STLs.

    Raises RuntimeError if neither the jaw pair nor a usable fallback is present —
    guessing here would produce a confident, wrong simulation.
    """
    names = (MANDIBLE, MAXILLA, PHARYNX, SINUS_L, SINUS_R,
             MOLAR_L, MOLAR_R, INCISOR_L, INCISOR_R)
    c = {n: _centroid(stl_dir, n) for n in names}
    notes = []

    if c[MAXILLA] is None or c[MANDIBLE] is None:
        raise RuntimeError(
            f"Cannot derive anatomical frame: need {MAXILLA}.stl and {MANDIBLE}.stl in {stl_dir}"
        )
    superior = _unit(c[MAXILLA] - c[MANDIBLE])
    notes.append("superior=maxilla-mandible")

    # ── lateral (measured first — see module docstring) ───────────────────
    if c[SINUS_L] is not None and c[SINUS_R] is not None:
        lateral_raw = c[SINUS_L] - c[SINUS_R]
        notes.append("lateral=sinusL-sinusR")
    elif c[MOLAR_L] is not None and c[MOLAR_R] is not None:
        lateral_raw = c[MOLAR_L] - c[MOLAR_R]
        notes.append("lateral=molar36-molar46")
    else:
        raise RuntimeError(
            "Cannot derive lateral axis: need both maxillary sinus STLs or both first-molar STLs"
        )
    lateral = _unit(lateral_raw - np.dot(lateral_raw, superior) * superior)

    # ── anterior = lateral × superior, sign fixed by the dental arch ──────
    anterior = _unit(np.cross(lateral, superior))
    front = [c[n] for n in (INCISOR_L, INCISOR_R) if c[n] is not None]
    back = [c[n] for n in (MOLAR_L, MOLAR_R) if c[n] is not None]
    if front and back:
        forward_cue = np.mean(front, axis=0) - np.mean(back, axis=0)
        notes.append("anterior sign from incisors vs molars")
    elif c[PHARYNX] is not None:
        forward_cue = c[MANDIBLE] - c[PHARYNX]      # pharynx sits behind the jaw
        notes.append("anterior sign from mandible vs pharynx")
    else:
        raise RuntimeError(
            "Cannot orient anterior axis: need incisor+molar STLs or a pharynx STL"
        )
    if np.dot(forward_cue, anterior) < 0:
        anterior = -anterior

    return AnatomicalFrame(superior, anterior, lateral, source="; ".join(notes))
