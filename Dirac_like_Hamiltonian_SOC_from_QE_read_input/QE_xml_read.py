from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pathlib

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QE_xml_read.py

Read k-points + lattice/reciprocal vectors from a Quantum ESPRESSO
data-file-schema.xml (from prefix.save/) and build a valley-truncated
list of valley-centered q-points suitable for a k·p / Dirac-like model.

Designed for workflows like:
  - Run SCF
  - Run NSCF on a dense 2D grid (e.g., 96x96x1) with nosym/noinv
  - Parse prefix.save/data-file-schema.xml
  - Truncate around K (and/or K') with radius k0 = alpha*(2π/a)
  - Return qx,qy in Å^-1 for your Dirac/BSE code param_list:
        param_list = [(Ev, Ec, qx, qy), ...]

Notes on units:
  - QE XML k_point and b-vectors are often given in "tpiba" units:
        1 tpiba = 2π / a(Å)  in Å^-1
    so if you pick k0 = alpha*(2π/a), then the cutoff radius is simply alpha in tpiba units.
"""

HA_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903


# -----------------------------
# XML helpers
# -----------------------------
def _strip_ns(tag: str) -> str:
    """Strip XML namespace if present: '{ns}tag' -> 'tag'."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _read_xml_scalars_vectors_stream(
    xml_path: str,
    want_tags: Tuple[str, ...] = ("a1", "b1", "b2"),
) -> Dict[str, np.ndarray]:
    """
    Stream-parse XML and extract vectors for tags like a1, b1, b2.
    Returns dict tag -> np.array([x,y,z]) as floats.
    """
    found: Dict[str, np.ndarray] = {}
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        tag = _strip_ns(elem.tag)
        if tag in want_tags and elem.text:
            vals = [float(x) for x in elem.text.split()]
            if len(vals) == 3:
                found[tag] = np.array(vals, dtype=float)
        # stop early if we have everything
        if all(t in found for t in want_tags):
            break
        elem.clear()
    return found


def read_kpoints_weights_stream(xml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stream-parse and return:
      kpts: (N,3) float array from <k_point> text
      wts:  (N,) float array from k_point attributes (weight)
    """
    kpts: List[List[float]] = []
    wts: List[float] = []

    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if _strip_ns(elem.tag) == "k_point":
            if elem.text is None:
                elem.clear()
                continue
            kpts.append([float(x) for x in elem.text.split()])
            wts.append(float(elem.attrib.get("weight", "nan")))
            elem.clear()

    if not kpts:
        raise RuntimeError("No <k_point> elements found. Are you using data-file-schema.xml?")
    return np.asarray(kpts, dtype=float), np.asarray(wts, dtype=float)


# -----------------------------
# Reciprocal-space geometry
# -----------------------------
def build_B2_from_b1b2(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Build 2x2 in-plane reciprocal matrix B2 in the *same* coordinate system
    as the k_point vectors (usually tpiba Cartesian).

    B2 columns are b1_xy and b2_xy:
        dk_cart = B2 @ dk_frac
    where dk_frac are coefficients in the (b1,b2) basis.
    """
    b1_xy = b1[:2]
    b2_xy = b2[:2]
    B2 = np.array([[b1_xy[0], b2_xy[0]],
                   [b1_xy[1], b2_xy[1]]], dtype=float)
    det = np.linalg.det(B2)
    if abs(det) < 1e-12:
        raise RuntimeError(f"Singular B2 (det={det}). Check lattice vectors in XML.")
    return B2


def min_image_delta(dk_xy: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """
    Minimum-image wrap of dk_xy in reciprocal space modulo (b1,b2).

    dk_xy: (...,2) Cartesian (tpiba) differences
    returns wrapped dk_xy in same coordinates, mapped to the nearest equivalent vector.
    """
    frac = np.linalg.solve(B2, dk_xy.T).T  # (...,2)
    frac -= np.round(frac)
    return (B2 @ frac.T).T


def K_cart_tpiba(B2: np.ndarray) -> np.ndarray:
    """
    Pick a conventional K point in the same (tpiba Cartesian) coordinates:
        K = (b1 + b2) / 3
    with b1,b2 from columns of B2.
    """
    b1 = B2[:, 0]
    b2 = B2[:, 1]
    return (b1 + b2) / 3.0


# -----------------------------
# Truncation + param_list
# -----------------------------
def read_aA_and_tpiba_to_Ainv(xml_path: str) -> Tuple[float, float]:
    """
    Read a1 (real-space lattice vector) from XML (in Bohr) and return:
      a_A: lattice constant a in Å (norm of a1)
      tpiba_to_Ainv: 2π/a  in Å^-1 per tpiba unit
    """
    vecs = _read_xml_scalars_vectors_stream(xml_path, want_tags=("a1",))
    if "a1" not in vecs:
        raise RuntimeError("Could not find <a1> in XML.")
    a_bohr = float(np.linalg.norm(vecs["a1"]))
    a_A = a_bohr * BOHR_TO_A
    tpiba_to_Ainv = 2.0 * math.pi / a_A
    return a_A, tpiba_to_Ainv


def truncate_valley_disk_tpiba(
    k_tpiba_xy: np.ndarray,
    B2: np.ndarray,
    alpha: float,
    valley: str = "K",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate kpoints to a disk around K or K' with radius alpha
    (in tpiba Cartesian units).

    Returns:
      q_tpiba_xy: (Nk,2) valley-centered minimum-image deltas (tpiba)
      mask:       (N,) boolean mask for kept points
      Kc:         (2,) center used (tpiba)
    """
    K = K_cart_tpiba(B2)
    if valley.lower() in ("k", "kplus", "k+", "k_plus"):
        Kc = K
    elif valley.lower() in ("kp", "kprime", "k'", "kminus", "k-", "k_minus"):
        Kc = -K
    else:
        raise ValueError("valley must be 'K' or 'Kp'/'Kprime'.")

    dk = k_tpiba_xy - Kc[None, :]
    dk_wrap = min_image_delta(dk, B2)  # minimum image
    r = np.linalg.norm(dk_wrap, axis=1)
    mask = r <= float(alpha)
    return dk_wrap[mask], mask, Kc[:2]


def read_b1_b2_tpiba(xml_path: str) -> np.ndarray:
    """
    Read b1,b2 vectors (in tpiba Cartesian) and return B2 (2x2)
    columns are b1_xy and b2_xy.
    """
    vecs = _read_xml_scalars_vectors_stream(xml_path, want_tags=("b1", "b2"))
    if "b1" not in vecs or "b2" not in vecs:
        raise RuntimeError("Could not find <b1> and <b2> in XML.")
    return build_B2_from_b1b2(vecs["b1"], vecs["b2"])


def build_param_list_from_qe_xml(
    xml_path: str,
    *,
    Ev: float = 0.0,
    Ec: float = 1.8,
    alpha: float = 0.10,
    valley: str = "K",
) -> Tuple[List[Tuple[float, float, float, float]], Dict[str, object]]:
    """
    Build param_list = [(Ev, Ec, qx_Ainv, qy_Ainv), ...] from QE XML.

    Uses valley-centered truncation with radius:
        k0 = alpha*(2π/a)
    and converts q from tpiba to Å^-1.
    """
    a_A, tpiba_to_Ainv = read_aA_and_tpiba_to_Ainv(xml_path)
    B2 = read_b1_b2_tpiba(xml_path)

    kpts, _ = read_kpoints_weights_stream(xml_path)
    k_xy = kpts[:, :2]  # tpiba Cartesian

    q_tpiba_xy, mask, Kc = truncate_valley_disk_tpiba(k_xy, B2, alpha=alpha, valley=valley)
    q_Ainv_xy = q_tpiba_xy * tpiba_to_Ainv  # Å^-1

    param_list = [(float(Ev), float(Ec), float(q[0]), float(q[1])) for q in q_Ainv_xy]

    info = {
        "a_A": a_A,
        "tpiba_to_Ainv": tpiba_to_Ainv,
        "Nk_total": int(k_xy.shape[0]),
        "Nk_kept": int(q_Ainv_xy.shape[0]),
        "valley": valley,
        "alpha": float(alpha),
        "K_center_tpiba_xy": [float(Kc[0]), float(Kc[1])],
    }
    return param_list, info


def compute_k_weight_from_qe_xml(xml_path: str) -> float:
    """
    k-space quadrature weight for 2D integrals with the convention:
        ∫ d^2k/(2π)^2  ≈  Σ_k  wk
    where wk = ΔA_k / (2π)^2 and ΔA_k = A_BZ / N_full.
    """
    a_A, tpiba_to_Ainv = read_aA_and_tpiba_to_Ainv(xml_path)
    vecs = _read_xml_scalars_vectors_stream(xml_path, want_tags=("b1", "b2"))
    b1 = vecs["b1"] * tpiba_to_Ainv
    b2 = vecs["b2"] * tpiba_to_Ainv
    A_BZ = np.linalg.norm(np.cross(b1, b2))  # Å^-2

    kpts, _ = read_kpoints_weights_stream(xml_path)
    Nfull = kpts.shape[0]
    deltaA = A_BZ / Nfull
    return float(deltaA / (2.0*np.pi)**2)


def extract_soc_splitting_at_valleys(xml_path: str, tol_ev=1e-6):
    """
    Returns dict with:
      fermi_eV,
      K:  {k_tpiba_xy, eig_eV, Ev_top, Ev_low, Ec_min, delta_v, delta_c}
      Kp: {same...}
    """
    B2 = read_b1_b2_tpiba(xml_path)
    K  = K_cart_tpiba(B2)
    Kp = -K

    fermi_ha = None

    best = {
        "K":  {"dist": 1e9, "k": None, "eigs_ha": None},
        "Kp": {"dist": 1e9, "k": None, "eigs_ha": None},
    }

    cur_k = None
    cur_eigs = None

    for _, el in ET.iterparse(xml_path, events=("end",)):
        tag = _strip_ns(el.tag)

        if tag == "fermi_energy" and (fermi_ha is None) and el.text:
            fermi_ha = float(el.text.strip())

        # collect within each ks_energies block
        if tag == "k_point" and el.text:
            txt = el.text.strip()
            if txt:
                cur_k = np.array([float(x) for x in txt.split()], float)[:2]
        elif tag == "eigenvalues" and el.text:
            txt = el.text.strip()
            if txt:
                cur_eigs = np.array([float(x) for x in txt.split()], float)

        # when a ks_energies block closes, process one k-point
        if tag == "ks_energies":
            if cur_k is not None and cur_eigs is not None:
                # distance to K / K' with min-image
                dK  = np.linalg.norm(min_image_delta((cur_k - K)[None, :],  B2)[0])
                dKp = np.linalg.norm(min_image_delta((cur_k - Kp)[None, :], B2)[0])

                if dK < best["K"]["dist"]:
                    best["K"].update(dist=dK, k=cur_k.copy(), eigs_ha=cur_eigs.copy())
                if dKp < best["Kp"]["dist"]:
                    best["Kp"].update(dist=dKp, k=cur_k.copy(), eigs_ha=cur_eigs.copy())

            cur_k = None
            cur_eigs = None

        el.clear()

    if fermi_ha is None:
        raise RuntimeError("Could not find fermi_energy in XML.")

    fermi_eV = fermi_ha * HA_TO_EV

    def analyze_one(eigs_ha):
        eigs_eV = eigs_ha * HA_TO_EV
        # valence = below fermi, conduction = above fermi
        v = eigs_eV[eigs_eV <= fermi_eV + tol_ev]
        c = eigs_eV[eigs_eV >= fermi_eV - tol_ev]
        v = np.sort(v)
        c = np.sort(c)

        Ev_top = v[-1]
        Ev_low = v[-2]
        Ec_min = c[0]
        # splittings
        delta_v = Ev_top - Ev_low
        delta_c = c[1] - c[0] if len(c) > 1 else np.nan
        return eigs_eV, Ev_top, Ev_low, Ec_min, delta_v, delta_c

    out = {"fermi_eV": fermi_eV}
    for key in ("K", "Kp"):
        eigs_eV, Ev_top, Ev_low, Ec_min, dv, dc = analyze_one(best[key]["eigs_ha"])
        out[key] = dict(
            dist_tpiba=best[key]["dist"],
            k_tpiba_xy=best[key]["k"].tolist(),
            Ev_top=Ev_top, Ev_low=Ev_low, Ec_min=Ec_min,
            delta_v=dv, delta_c=dc,
        )
    return out


# -----------------------------
# SOC detection helpers (NEW)
# -----------------------------
def _find_key_recursive(obj, needle: str):
    """Return list of values for keys matching `needle` anywhere in nested dict/list."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == needle:
                out.append(v)
            out.extend(_find_key_recursive(v, needle))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_find_key_recursive(v, needle))
    return out


def read_qe_calculation_flags(xml_path: str) -> Dict[str, object]:
    """
    Best-effort read of key flags from data-file-schema.xml:
      noncolin / lspinorb / spinorbit / lsda / nspin

    QE XML schema varies by version; this function is intentionally robust:
      - quick text scan for <tag>true</tag> or tag="true"
      - then structured scan to pick up nspin if present
    """
    raw = pathlib.Path(xml_path).read_text(errors="ignore").lower()

    def _has_true(key: str) -> bool:
        return (f"<{key}>true</{key}>" in raw) or (f'{key}="true"' in raw)

    flags = {
        "noncolin": _has_true("noncolin"),
        "lspinorb": _has_true("lspinorb") or _has_true("spinorbit") or _has_true("spin_orbit"),
        "lsda": _has_true("lsda"),
        "nspin": None,
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def _get_text_first(tagname: str):
            for el in root.iter():
                if _strip_ns(el.tag).lower() == tagname.lower() and el.text:
                    return el.text.strip()
            return None

        for k in ("noncolin", "lspinorb", "spinorbit", "spin_orbit", "lsda"):
            val = _get_text_first(k)
            if val is not None and val.lower() in ("true", "false"):
                if k in ("spinorbit", "spin_orbit"):
                    flags["lspinorb"] = (val.lower() == "true") or flags["lspinorb"]
                else:
                    flags[k] = (val.lower() == "true")

        nspin_txt = _get_text_first("nspin")
        if nspin_txt is not None:
            try:
                flags["nspin"] = int(float(nspin_txt))
            except Exception:
                flags["nspin"] = nspin_txt
    except Exception:
        pass

    flags["soc_requested"] = bool(flags["noncolin"] and flags["lspinorb"])
    return flags


def detect_soc_from_qe_xml(xml_path: str, split_tol_ev: float = 1e-3) -> Dict[str, object]:
    """
    Decide whether SOC is enabled.

    Priority:
      (A) XML flags: noncolin && lspinorb  -> SOC on
      (B) fallback: band splitting at K (delta_v) > split_tol_ev

    Returns:
      soc_on, method, flags, delta_v_K, delta_c_K
    """
    flags = read_qe_calculation_flags(xml_path)
    if flags.get("soc_requested", False):
        return {
            "soc_on": True,
            "method": "xml_flags",
            "flags": flags,
            "delta_v_K": None,
            "delta_c_K": None,
        }

    try:
        spl = extract_soc_splitting_at_valleys(xml_path)
        dv = float(spl["K"]["delta_v"])
        dc = float(spl["K"]["delta_c"]) if spl["K"]["delta_c"] is not None else float("nan")
        soc_on = (abs(dv) > split_tol_ev)
        return {
            "soc_on": bool(soc_on),
            "method": "band_splitting",
            "flags": flags,
            "delta_v_K": dv,
            "delta_c_K": dc,
        }
    except Exception as e:
        return {
            "soc_on": False,
            "method": "unknown",
            "flags": flags,
            "delta_v_K": None,
            "delta_c_K": None,
            "note": f"Could not evaluate band splittings: {e}",
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xml", help="path to data-file-schema.xml")
    p.add_argument("--Ev", type=float, default=0.0)
    p.add_argument("--Ec", type=float, default=1.8)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--valley", type=str, default="K")
    args = p.parse_args()

    plist, info = build_param_list_from_qe_xml(
        args.xml, Ev=args.Ev, Ec=args.Ec, alpha=args.alpha, valley=args.valley
    )
    print(json.dumps(info, indent=2))
    print("param_list length =", len(plist))


if __name__ == "__main__":
    main()