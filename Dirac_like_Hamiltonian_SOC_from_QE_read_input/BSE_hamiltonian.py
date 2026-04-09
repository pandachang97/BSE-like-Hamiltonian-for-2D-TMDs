
# In this file, we build up BSE Hamiltonian based on Dirac-like Hamiltonian.
"""
Build BSE-like Hamiltonians for two valleys from Dirac-like 2x2 series.

Workflow:
1) Provide a param_list of (Ev, Ec, kx, ky) tuples or dicts and build the series
   with K_P_dirac.build_two_valley_dirac_series(..., a=..., t=...).
2) For each valley we build an N x N BSE-like Hamiltonian:
      H_BSE[i,i] = Δ_i  (energy gap at k_i, i.e., eigvals[1]-eigvals[0])
      H_BSE[i,j] = W(i, j) for i != j    # you supply W via a callable.
3) Conjugation rule for off-diagonals:
      K_plus : H[i,j]=W(i,j),   H[j,i]=conj(W(i,j))
      K_minus: H[i,j]=conj(W),  H[j,i]=W
You can supply different kernels per valley (kernel_plus, kernel_minus).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# --- import your Dirac utilities ---
# --- import your Dirac utilities ---
from K_P_dirac import (
    build_two_valley_dirac_series,
    KPItem,          # holds H, eigvals, eigvecs, params=(Ev,Ec,kx,ky), meta={...}
    ValleySeries,    # items, H_stack, eigvals_stack, eigvecs_stack
)

ArrayC = NDArray[np.complex128]
ArrayF = NDArray[np.float64]

# Kernel type: given two node indices and the valley items, returns W_ij (complex, eV)
KernelFunc = Callable[[int, int, Sequence[KPItem]], complex]

# -------------------- RPA-like Coulomb kernel (flexible, in eV) -------------------- #

def coulomb_rpa_kernel_flexible(
    *,
    chi_value: float,           # your number, either length or inverse-length
    chi_units: str,             # "A" (Å), "bohr", "A^-1", or "bohr^-1"
    chi_is_inverse: bool,       # False if chi_value is a length; True if inverse-length
    k_units: str = "A^-1",      # units of your k-grid: "A^-1" or "bohr^-1"
    k_eps: float = 1e-12,
) -> KernelFunc:
    """
    W(k) = 2π e^2 / [ k * (1 + 2π χ_len k) ]  (computed in a.u., returned in eV)
    where:
      - If chi_is_inverse = False, chi_value is χ_len (length).
      - If chi_is_inverse = True,  chi_value is χ_inv (inverse length) such that
            1 + 2π χ_len k  ==  1 + k / χ_inv.

    k is the momentum transfer |k_i - k_j|.
    """

    EH_TO_EV = 27.211386245988          # Hartree → eV
    A_TO_BOHR = 1.889726124565062       # Å → Bohr
    AINV_TO_BOHRINV = 0.529177210903       # Å^-1 -> Bohr^-1  (IMPORTANT!)

    # normalize chi to *Bohr* or *Bohr^-1*
    if not chi_is_inverse:
        # chi_value is length
        if chi_units.lower() in ("a", "ang", "angstrom"):
            chi_len_bohr = chi_value * A_TO_BOHR
        elif chi_units.lower() in ("bohr", "a0"):
            chi_len_bohr = float(chi_value)
        else:
            raise ValueError("chi_units must be 'A'/'angstrom' or 'bohr' for length mode")
        # closure uses length form: 1 + 2π χ_len k
        def denom_factor(k_bohr: float) -> float:
             # 1 + 2π χ_len k   (π does NOT become 1 in atomic units)
            return 1.0 + 2.0 * np.pi * chi_len_bohr * k_bohr
    else:
        # chi_value is inverse length (χ_inv)
        if chi_units.lower() in ("a^-1", "ang^-1", "angstrom^-1"):
            chi_inv_bohr = chi_value * AINV_TO_BOHRINV
        elif chi_units.lower() in ("bohr^-1", "a0^-1"):
            chi_inv_bohr = float(chi_value)
        else:
            raise ValueError("chi_units must be 'A^-1'/'angstrom^-1' or 'bohr^-1' for inverse-length mode")
        # closure uses inverse-length form: 1 + k / χ_inv
        def denom_factor(k_bohr: float) -> float:
            return 1.0 + (k_bohr / chi_inv_bohr)

    # normalize how to convert k to Bohr^-1
    if k_units.lower() in ("a^-1", "ang^-1", "angstrom^-1"):
        def k_to_bohr(k: float) -> float:
            return k * AINV_TO_BOHRINV
    elif k_units.lower() in ("bohr^-1", "a0^-1"):
        def k_to_bohr(k: float) -> float:
            return float(k)
    else:
        raise ValueError("k_units must be 'A^-1' or 'bohr^-1'")

    def kernel(i: int, j: int, items: Sequence[KPItem]) -> float:
        # momentum transfer |k_i - k_j|
        _, _, kxi, kyi = items[i].params
        _, _, kxj, kyj = items[j].params
        dkx = float(kxi - kxj)
        dky = float(kyi - kyj)
        k_mag = (dkx**2 + dky**2) ** 0.5
        if k_mag < k_eps:
            return 0.0 

        k_bohr = k_to_bohr(k_mag)
        denom = k_bohr * denom_factor(k_bohr)      # k * (1 + 2π χ_len k) or k * (1 + k/χ_inv)
        W_hartree = ( 2.0 * np.pi ) / denom          # e^2 = 1 in a.u.
        return float(W_hartree * EH_TO_EV)       # return eV

    return kernel

# ----------------------------- Helper utilities --------------------------- #

def energy_gaps_from_items(items: Sequence[KPItem]) -> ArrayF:
    """
    Compute Δ_i = E_c(i) - E_v(i) at each k_i.
    Assumes np.linalg.eigh ordering: eigvals[0] <= eigvals[1].
    Returns a real array (tiny imaginary noise is discarded).
    """
    N = len(items)
    gaps = np.empty(N, dtype=np.float64)
    for i, it in enumerate(items):
        gaps[i] = (it.eigvals[1] - it.eigvals[0]).real
    return gaps


def k_grid_from_items(items: Sequence[KPItem]) -> ArrayF:
    """
    Extract k = (kx, ky) for convenience if your kernel needs |k_i - k_j|.
    Shape: (N, 2).  Values are float64.
    """
    N = len(items)
    K = np.empty((N, 2), dtype=np.float64)
    for i, it in enumerate(items):
        _, _, kx, ky = it.params
        K[i, 0] = kx
        K[i, 1] = ky
    return K

def k_complex_vector_from_items(items: Sequence[KPItem]) -> np.ndarray:
    """
    Return the complex k-grid vector z_i stored by K_P_dirac for one valley:
        z_i = kx ± i ky  (already valley-dependent)
    Requires KPItem.meta['k_complex'] to be present.
    """
    z = []
    for it in items:
        if it.meta is None or "k_complex" not in it.meta:
            raise ValueError(
                "KPItem.meta['k_complex'] not found. "
                "Rebuild the series with K_P_dirac.build_two_valley_dirac_series "
                "(the version that stores 'k_complex' in meta)."
            )
        z.append(it.meta["k_complex"])
    return np.asarray(z, dtype=np.complex128)


def phi_angles_from_items_meta(
    items: Sequence[KPItem],
    *,
    normalize_to_2pi: bool = True,
) -> ArrayF:
    """
    Compute φ_i = arg(k_complex_i) for one valley, using the pre-stored complex k-grid.
    If normalize_to_2pi is True, map to [0, 2π); otherwise uses (-π, π].
    """
    z = k_complex_vector_from_items(items)   # complex array (N,)
    phi = np.angle(z)                        # in (-π, π]
    if normalize_to_2pi:
        phi = np.mod(phi, 2.0 * np.pi)       # to [0, 2π)
    return phi.astype(np.float64, copy=False)


def phi_angles_two_valleys(
    series: Dict[str, ValleySeries],
    *,
    normalize_to_2pi: bool = True,
) -> Tuple[ArrayF, ArrayF]:
    """
    Convenience wrapper to get:
        phi_angle_plus  = arg(kx + i ky) for K_plus
        phi_angle_minus = arg(kx - i ky) for K_minus
    Both are read from KPItem.meta['k_complex'].
    """
    phi_angle_plus  = phi_angles_from_items_meta(series["K_plus"].items,
                                                 normalize_to_2pi=normalize_to_2pi)
    phi_angle_minus = phi_angles_from_items_meta(series["K_minus"].items,
                                                 normalize_to_2pi=normalize_to_2pi)
    return phi_angle_plus, phi_angle_minus


def build_bse_matrix_with_phase(
    items: Sequence[KPItem],
    kernel: KernelFunc,
    phi: ArrayF,                         # length-N, phase angles for this valley
     tau: int,      # +1 for K_plus, -1 for K_minus
    *,
    k_weight=1.0,              # scalar or length-N array
) -> ArrayC:
    """
    Build an N×N BSE-like Hamiltonian for one valley:
      H[i,i] = Δ_i
      For i<j:
         Z_ij = W(i,j) * exp(-i * (phi[i] - phi[j]))
         if upper_uses_conjugate is False (K_plus):
             H[i,j] = Z_ij;      H[j,i] = conj(Z_ij)
         else (K_minus):
             H[i,j] = conj(Z_ij); H[j,i] = Z_ij
    Ensures Hermiticity.
    """
    N = len(items)
    if N == 0:
        raise ValueError("No Dirac-like Hamiltonians found (N=0).")

    gaps = energy_gaps_from_items(items)
    H = np.zeros((N, N), dtype=np.complex128)
    np.fill_diagonal(H, gaps.astype(np.complex128, copy=False))
     # allow scalar or per-k weights
    if np.isscalar(k_weight):
        w = None
        w_scalar = float(k_weight)
    else:
        w = np.asarray(k_weight, dtype=float)
        if w.shape != (N,):
            raise ValueError("k_weight must be scalar or shape (N,)")

    for i in range(N):
        for j in range(i+1, N):
            Wij = -complex(kernel(i, j, items))      # attractive direct term
            dphi = phi[i] - phi[j]
            phase = np.exp(-1j * tau * dphi)         # <-- tau enters here

            wj = w_scalar if w is None else w[j]
            Zij = wj * Wij * phase                   # <-- k-space weight

            H[i, j] = Zij
            H[j, i] = np.conjugate(Zij)              # Hermitian by construction
    return H


# ----------------------------- Public API --------------------------------- #

@dataclass
class BSEOutput:
    """
    Container for one valley’s BSE-like Hamiltonian + helpers.
    """
    H_BSE: ArrayC            # (N, N)
    gaps: ArrayF             # (N,)  Δ_i for diagnostics
    k_grid: ArrayF           # (N,2) kx, ky for convenience


def build_bse_two_valleys_from_series(
    series: Dict[str, ValleySeries],
    kernel_plus: KernelFunc | None = None,
    kernel_minus: KernelFunc | None = None,
    *,
    k_weight=1.0,   # <-- ADD THIS (scalar or (N,) array)
    # Defaults reflect your note that χ may be provided in Å^-1
    chi_default: float = 6.60,
    chi_units_default: str = "A",   # "A" if you have χ as a length instead
    chi_is_inverse_default: bool = False,
    k_units_default: str = "A^-1",
    k_eps_default: float = 1e-12,
) -> Dict[str, BSEOutput]:
    """
    Build BSE-like Hamiltonians for both valleys from a prepared two-valley series.

    If kernel_plus (or kernel_minus) is None, we default to the Coulomb–RPA kernel:
      W(k) = 2π e^2 / [ k (1 + 2π χ k) ]  (computed in a.u., returned in eV)
    with χ interpreted according to (chi_units_default, chi_is_inverse_default).
    """
    # resolve default kernels to your W(k)
    if kernel_plus is None:
        kernel_plus = coulomb_rpa_kernel_flexible(
            chi_value=chi_default,
            chi_units=chi_units_default,
            chi_is_inverse=chi_is_inverse_default,
            k_units=k_units_default,
            k_eps=k_eps_default,
        )
    if kernel_minus is None:
        kernel_minus = kernel_plus  # reuse unless user provided a distinct one

    plus_items  = series["K_plus"].items
    minus_items = series["K_minus"].items
    
    # get valley phase angles from stored k_complex
    phi_plus, phi_minus = phi_angles_two_valleys(series, normalize_to_2pi=False)

    # K_plus: H[i,j]=W; H[j,i]=W*
    H_plus = build_bse_matrix_with_phase(
    series["K_plus"].items, kernel_plus, phi_plus, tau=+1, k_weight=k_weight)
    # K_minus: H[i,j]=W*; H[j,i]=W
    H_minus = build_bse_matrix_with_phase(
    series["K_minus"].items, kernel_minus, phi_minus, tau=-1, k_weight=k_weight)

    gaps_p = energy_gaps_from_items(plus_items)
    gaps_m = energy_gaps_from_items(minus_items)
    Kp = k_grid_from_items(plus_items)
    Km = k_grid_from_items(minus_items)

    return {
        "K_plus":  BSEOutput(H_BSE=H_plus, gaps=gaps_p, k_grid=Kp),
        "K_minus": BSEOutput(H_BSE=H_minus, gaps=gaps_m, k_grid=Km),
    }
