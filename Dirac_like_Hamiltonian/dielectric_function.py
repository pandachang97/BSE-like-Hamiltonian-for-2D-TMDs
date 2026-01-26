
# In this file, we diagonalize the BSE Hamiltonian and calculate the osicllator strength (imaginary part of the dielectric function).
 
"""
Diagonalize two-valley BSE matrices and compute the imaginary dielectric function.

Outputs include:
- eigvals/ eigvecs for K_plus and K_minus (stored separately as NumPy arrays)
- per-valley spectra and the total ε2(ω)

Assumes BSE matrices are built in eV (from BSE_hamiltonian.build_bse_two_valleys_from_series).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

from K_P_dirac import ValleySeries
from BSE_hamiltonian import (
    BSEOutput,
    build_bse_two_valleys_from_series,  # resolves kernel defaults internally
)

ArrayF = NDArray[np.float64]
ArrayC = NDArray[np.complex128]


# ----------------------- Eigen-solve & oscillator pieces ----------------------- #

def diagonalize_bse_matrix(H: ArrayC) -> Tuple[ArrayF, ArrayC]:
    """
    Hermitian eigen-solve: returns (eigenvalues asc, eigenvectors as columns).
    """
    w, v = np.linalg.eigh(H)
    return w.astype(np.float64, copy=False), v.astype(np.complex128, copy=False)


@dataclass
class BSEEigens:
    eigvals: ArrayF   # (N,)
    eigvecs: ArrayC   # (N,N) columns are eigenvectors


def diagonalize_two_valley_bse(bse: Dict[str, BSEOutput]) -> Dict[str, BSEEigens]:
    """
    Diagonalize BSE matrices for both valleys and return separate arrays.
    """
    Hp = bse["K_plus"].H_BSE
    Hm = bse["K_minus"].H_BSE
    Ep, Vp = diagonalize_bse_matrix(Hp)
    Em, Vm = diagonalize_bse_matrix(Hm)
    return {
        "K_plus":  BSEEigens(eigvals=Ep, eigvecs=Vp),
        "K_minus": BSEEigens(eigvals=Em, eigvecs=Vm),
    }

# ---------- Pull φk from stored k_complex (no re-deriving) ----------

def _kcomplex_vector_from_items(items: Sequence) -> np.ndarray:
    z = []
    for it in items:
        if it.meta is None or "k_complex" not in it.meta:
            raise ValueError("KPItem.meta['k_complex'] missing; rebuild series in K_P_dirac.")
        z.append(it.meta["k_complex"])
    return np.asarray(z, dtype=np.complex128)

def phi_angles_two_valleys_from_series(series: Dict[str, ValleySeries]) -> Tuple[ArrayF, ArrayF]:
    z_plus  = _kcomplex_vector_from_items(series["K_plus"].items)
    z_minus = _kcomplex_vector_from_items(series["K_minus"].items)
    # angles in (-π, π]; map to [0, 2π) if you prefer—physics here uses raw angle
    phi_plus  = np.angle(z_plus).astype(np.float64, copy=False)
    phi_minus = np.angle(z_minus).astype(np.float64, copy=False)
    return phi_plus, phi_minus

# ---------- Your phase-based oscillator strengths ----------

def oscillator_strengths_circular_right(
    evecs: ArrayC,
    valley_series: ValleySeries,
    tau: int,
    *,
    zero_guard: float = 1e-12,
) -> ArrayF:
    """
    Right-circular oscillator strengths for one valley:

        D_S = Σ_i W_i * C_{iS},      strengths_S = |D_S|^2

    with
        W_i = exp(-i * tau * φ_i) * (1 + tau * Eg_i / Ek_i)

    where
        φ_i   = arg(k_complex_i) read from KPItem.meta["k_complex"]  (already valley-specific)
        Eg_i  = |Ec - Ev| using the diagonal inputs (from params)
        Ek_i  = |eigval_high - eigval_low| at that k-point (from item.eigvals)

    tau = +1 for K_plus, -1 for K_minus.
    """
    # angles from stored complex k (no re-derivation)
    z = []
    Eg = []
    Ek = []
    items = valley_series.items
    for it in items:
        # k phase
        if it.meta is None or "k_complex" not in it.meta:
            raise ValueError("KPItem.meta['k_complex'] missing; rebuild series in K_P_dirac.")
        z.append(it.meta["k_complex"])
        # Eg from input diagonals
        Ev, Ec, _, _ = it.params
        Eg.append(abs(Ec - Ev))
        # Ek from eigenvalues of the 2x2 Dirac Hamiltonian at this k
        lam = np.sort(np.real_if_close(it.eigvals))
        Ek.append(abs(lam[1] - lam[0]))

    z  = np.asarray(z,  dtype=np.complex128)
    Eg = np.asarray(Eg, dtype=np.float64)
    Ek = np.asarray(Ek, dtype=np.float64)

    phi = np.angle(z)  # (-pi, pi]
    # W_i = e^{-i τ φ_i} * (1 + τ * Eg/Ek)
    ratio = Eg / np.maximum(Ek, zero_guard)
    W = np.exp(-1j * tau * phi) * (1.0 + tau * ratio)

    # D_S = Σ_i W_i C_{iS}  (evecs: shape N×N with columns = eigenvectors)
    D = W @ evecs
    return (np.abs(D) ** 2).astype(np.float64, copy=False)

def dielectric_spectrum(
    energies: ArrayF,          # Ω_S (eV), shape (S,)
    strengths: ArrayF,         # |D_S|^2 (or your circular version), shape (S,)
    omega_grid: ArrayF,        # (M,) eV
    eta: float = 0.05,         # HWHM
    prefactor: float = 1.0,
    include_frequency_factor: bool = True,
    zero_guard: float = 1e-12,
    broadening: str = "lorentzian",   # "lorentzian" or "gaussian"
) -> ArrayF:
    """
    ε2(ω) = Σ_S w_S * L(ω-Ω_S),
      where w_S = |D_S|^2 / Ω_S^2 if include_frequency_factor else |D_S|^2

    Broadening:
      - "lorentzian": L(x) = (η/π) / (x^2 + η^2)
      - "gaussian"  : L(x) = [1/(√(2π) σ)] * exp( -x^2/(2σ^2) ) with σ = η / √(2 ln 2)
                      (η is treated as the Gaussian HWHM)
    """
    if include_frequency_factor:
        weights = strengths / np.maximum(energies * energies, zero_guard)
    else:
        weights = strengths

    x = omega_grid.reshape(-1, 1) - energies.reshape(1, -1)  # (M,S)

    b = broadening.strip().lower()
    if b == "lorentzian":
        L = (eta / np.pi) / (x * x + eta * eta)
    elif b == "gaussian":
        sigma = max(eta / np.sqrt(2.0 * np.log(2.0)), zero_guard)  # HWHM→σ
        L = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-(x * x) / (2.0 * sigma * sigma))
    else:
        raise ValueError("broadening must be 'lorentzian' or 'gaussian'")

    return (prefactor * (L @ weights.reshape(-1, 1)).ravel()).astype(np.float64, copy=False)

# ----------------------- High-level wrapper ----------------------- #

@dataclass
class DielectricResult:
    omega: ArrayF
    eps2_total: ArrayF
    eps2_K_plus: ArrayF
    eps2_K_minus: ArrayF
    exc_energies_K_plus: ArrayF
    exc_strengths_K_plus: ArrayF
    exc_energies_K_minus: ArrayF
    exc_strengths_K_minus: ArrayF
    eigvals_K_plus: ArrayF
    eigvecs_K_plus: ArrayC
    eigvals_K_minus: ArrayF
    eigvecs_K_minus: ArrayC

def compute_dielectric_for_two_valleys(
    series: Dict[str, ValleySeries],
    *,
    polarization: str = "right",   # "right" or "left"
    eta: float = 0.05,
    omega_min: float = 0.0,
    omega_max: float = 4.0,
    omega_points: int = 2000,
    prefactor: float = 1.0,
    include_frequency_factor: bool = True,  # keep your 1/E^2 option
    broadening: str = "gaussian",   # "lorentzian" or "gaussian"
) -> DielectricResult:
    """
    1) Build BSE (Hp, Hm) using your BSE_hamiltonian defaults (phase-aware kernel).
    2) Diagonalize each valley; store eigenpairs as arrays.
    3) Compute *right-circular* oscillator strengths:
           W_i^(τ) = exp(-i τ φ_i) * (1 + τ * Eg_i / Ek_i),
       with τ=+1 for K_plus, τ=-1 for K_minus.
       (Left-circular is handled by swapping valley spectra below.)
    4) Build ε2(ω) with Lorentzian broadening.

    polarization:
        "right" -> return (eps2_K_plus, eps2_K_minus) as computed.
        "left"  -> swap the per-valley spectra in the output.
    """
    pol = polarization.strip().lower()
    if pol not in ("right", "left"):
        raise ValueError("polarization must be 'right' or 'left'")

    # --- BSE matrices ---
    bse = build_bse_two_valleys_from_series(series)

    # --- Eigenpairs ---
    eigs = diagonalize_two_valley_bse(bse)
    Ep, Vp = eigs["K_plus"].eigvals,  eigs["K_plus"].eigvecs
    Em, Vm = eigs["K_minus"].eigvals, eigs["K_minus"].eigvecs

    # --- Right-circular oscillator strengths  ---
    Sp = oscillator_strengths_circular_right(Vp, series["K_plus"],  tau=+1)
    Sm = oscillator_strengths_circular_right(Vm, series["K_minus"], tau=-1)

    # --- Spectrum on a uniform energy grid ---
    omega = np.linspace(omega_min, omega_max, omega_points, dtype=np.float64)
    eps2_p = dielectric_spectrum(
        Ep, Sp, omega, eta=eta, prefactor=prefactor,
        include_frequency_factor=include_frequency_factor,
        broadening=broadening,
    )
    eps2_m = dielectric_spectrum(
        Em, Sm, omega, eta=eta, prefactor=prefactor,
        include_frequency_factor=include_frequency_factor,
        broadening=broadening,
    )

    # For left-circular, swap the per-valley spectra in the OUTPUT
    if pol == "left":
        eps2_p, eps2_m = eps2_m, eps2_p

    eps2_total = eps2_p + eps2_m

    return DielectricResult(
        omega=omega,
        eps2_total=eps2_total,
        eps2_K_plus=eps2_p,      # swapped if polarization == "left"
        eps2_K_minus=eps2_m,     # swapped if polarization == "left"
        exc_energies_K_plus=Ep,
        exc_strengths_K_plus=Sp,
        exc_energies_K_minus=Em,
        exc_strengths_K_minus=Sm,
        eigvals_K_plus=Ep,
        eigvecs_K_plus=Vp,
        eigvals_K_minus=Em,
        eigvecs_K_minus=Vm,
    )


def plot_dielectric(res, show_per_valley=True, title="Dielectric Spectrum"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(res.omega, res.eps2_total, label="Total ε2(ω)")
    if show_per_valley:
        ax.plot(res.omega, res.eps2_K_plus, label="K₊ valley")
        ax.plot(res.omega, res.eps2_K_minus, label="K₋ valley")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Im ε(ω) (arb. units)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
