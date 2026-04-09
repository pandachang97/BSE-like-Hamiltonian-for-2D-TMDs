
# In this file, we define the Dirac-like Hamiltonian for the K.P model.
# The parameters should be read from the parameters.json file.
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.complex128]


ArrayC = NDArray[np.complex128]
ArrayF = NDArray[np.float64]

@dataclass
class KPItem:
    """One 2x2 Hamiltonian + eigensystem + its parameters."""
    H: ArrayC                 # (2,2)
    eigvals: ArrayC           # (2,)
    eigvecs: ArrayC           # (2,2) columns are eigenvectors
    params: Tuple[float, float, float, float]  # (Ev, Ec, kx, ky)
    meta: dict | None = None                  # optional extra info (a, t, k_complex, valley)

@dataclass
class ValleySeries:
    """Container for one valley’s entire series and convenient stacks."""
    items: List[KPItem]
    H_stack: ArrayC          # (N,2,2)
    eigvals_stack: ArrayC    # (N,2)
    eigvecs_stack: ArrayC    # (N,2,2)

def k_complex(kx: float, ky: float, valley: str) -> complex:
    """Valley-dependent complex k-grid."""
    if valley == "K_plus":
        return complex(kx, ky)    # kx + i ky
    elif valley == "K_minus":
        return complex(-kx, ky)   # kx + i ky
    raise ValueError("valley must be 'K_plus' or 'K_minus'")

def _dirac_like_matrix_valley(
    Ev: float, Ec: float, kx: float, ky: float, valley: str,
    a: float = 1.0, t: float = 1.0
) -> ArrayC:
    """
    Build 2x2 Dirac-like H with off-diagonal = (kx ± i ky) * a * t, per valley.
    Diagonals are Ev and Ec (leave your physics choice here).
    """
    off = k_complex(kx, ky, valley) * (a * t)
    return np.array(
        [[Ev, off],
         [np.conjugate(off), Ec]],
        dtype=np.complex128
    )


def _stack_from_items(items: List[KPItem]) -> Tuple[ArrayC, ArrayC, ArrayC]:
    N = len(items)
    H_stack       = np.empty((N, 2, 2), dtype=np.complex128)
    eigvals_stack = np.empty((N, 2),     dtype=np.complex128)
    eigvecs_stack = np.empty((N, 2, 2),  dtype=np.complex128)
    for i, it in enumerate(items):
        H_stack[i]       = it.H
        eigvals_stack[i] = it.eigvals
        eigvecs_stack[i] = it.eigvecs
    return H_stack, eigvals_stack, eigvecs_stack

def build_two_valley_dirac_series(
    params: Sequence[Union[Tuple[float, float, float, float], dict]],
    *,
    a: float = 1.0,    # lattice constant (same units as 1/k)
    t: float = 1.0,    # hopping/scale (energy)
) -> Dict[str, ValleySeries]:
    """
    Build and diagonalize series for both valleys with off-diagonal = k_complex * a * t.

    params: sequence of (Ev, Ec, kx, ky) tuples or dicts with those keys.
    a: lattice constant (if k is in Å^-1, give a in Å so a*k is dimensionless)
    t: energy scale multiplying a*k.
    """
    plus_items: List[KPItem]  = []
    minus_items: List[KPItem] = []

    for p in params:
        if isinstance(p, dict):
            Ev, Ec, kx, ky = float(p["Ev"]), float(p["Ec"]), float(p["kx"]), float(p["ky"])
        else:
            Ev, Ec, kx, ky = map(float, p)

        # K_plus
        Hp = _dirac_like_matrix_valley(Ev, Ec, kx, ky, "K_plus", a=a, t=t)
        wp, vp = np.linalg.eigh(Hp)
        plus_items.append(
            KPItem(
                H=Hp,
                eigvals=wp.astype(np.complex128, copy=False),
                eigvecs=vp.astype(np.complex128, copy=False),
                params=(Ev, Ec, kx, ky),
                meta={"valley": "K_plus", "a": a, "t": t, "k_complex": k_complex(kx, ky, "K_plus")}
            )
        )

        # K_minus
        Hm = _dirac_like_matrix_valley(Ev, Ec, kx, ky, "K_minus", a=a, t=t)
        wm, vm = np.linalg.eigh(Hm)
        minus_items.append(
            KPItem(
                H=Hm,
                eigvals=wm.astype(np.complex128, copy=False),
                eigvecs=vm.astype(np.complex128, copy=False),
                params=(Ev, Ec, kx, ky),
                meta={"valley": "K_minus", "a": a, "t": t, "k_complex": k_complex(kx, ky, "K_minus")}
            )
        )

    HpS, wpS, vpS = _stack_from_items(plus_items)
    HmS, wmS, vmS = _stack_from_items(minus_items)

    return {
        "K_plus":  ValleySeries(items=plus_items,  H_stack=HpS, eigvals_stack=wpS, eigvecs_stack=vpS),
        "K_minus": ValleySeries(items=minus_items, H_stack=HmS, eigvals_stack=wmS, eigvecs_stack=vmS),
    }


