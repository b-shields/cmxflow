"""Numba-compiled inner kernel for the empirical (Vinardo) scoring function.

Isolated in its own module so that:

* ``numba`` is imported only when scoring actually runs (this module is
  lazy-imported from ``score.py``), keeping ~0.3 s of numba import off the
  widely-imported ``score.py`` path and therefore off CLI startup.
* the kernel lives at module scope, which is required for numba's
  ``cache=True`` on-disk artifact (closures have no cache locator), so
  ``make_parallel`` workers reload the compiled kernel from ``__pycache__``
  instead of each recompiling.

The kernel reproduces the math of ``score._pair_term_sums_and_grad`` plus the
gradient scatter, fused into a single pass over the neighbor-pair list. The
per-pair gather (protein coordinate, radius sum, hydrophobic / H-bond pair
masks) is done *inside* the loop by indexing the cached protein arrays directly,
which avoids the ~200 µs of numpy fancy-index temporaries the caller would
otherwise allocate each call. It returns the four *unweighted* term sums (so
callers can still build ``ScoreComponents``) and the per-ligand-atom gradient.

``fastmath=True`` is enabled for the most performant path. Combined with
sequential (vs. numpy pairwise) summation this drifts the result by ~1e-10 from
the numpy path -- negligible against the smina calibration MAE (~0.2 kcal/mol),
and may nudge a golden pose within its flat L-BFGS-B basin (accepted: regenerate
fixtures if needed).
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def score_grad_pairs(
    coords: np.ndarray,
    protein_coords: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    lig_radii: np.ndarray,
    prot_radii: np.ndarray,
    lig_hydro: np.ndarray,
    prot_hydro: np.ndarray,
    lig_don: np.ndarray,
    lig_acc: np.ndarray,
    prot_don: np.ndarray,
    prot_acc: np.ndarray,
    w_gauss1: float,
    w_repulsion: float,
    w_hydrophobic: float,
    w_hbond: float,
    gauss1_offset: float,
    gauss1_width: float,
    hydro_good: float,
    hydro_bad: float,
    hbond_good: float,
    core_cutoff: float,
    prot_weight: np.ndarray,
    inv_divisor: float,
) -> tuple[float, float, float, float, np.ndarray]:
    """Vinardo term sums + per-atom gradient over a fixed neighbor-pair list.

    Each entry ``p`` is a (ligand atom ``i_idx[p]``, protein atom ``j_idx[p]``)
    pair within the interaction cutoff. The surface distance is
    ``|coords[i] - protein_coords[j]| - lig_radii[i] - prot_radii[j]``; pair
    types are derived from the per-atom hydrophobic / donor / acceptor masks.

    Args:
        coords: Ligand heavy-atom coordinates (n_lig, 3), C-contiguous.
        protein_coords: Protein coordinates (n_prot, 3).
        i_idx: Ligand atom index per pair (n_pairs,), int64.
        j_idx: Protein atom index per pair (n_pairs,), int64.
        lig_radii, prot_radii: Per-atom Vinardo radii.
        lig_hydro, prot_hydro: Per-atom hydrophobic masks.
        lig_don, lig_acc, prot_don, prot_acc: Per-atom H-bond donor/acceptor masks.
        w_gauss1, w_repulsion, w_hydrophobic, w_hbond: Term weights.
        gauss1_offset, gauss1_width: Gaussian center/width.
        hydro_good, hydro_bad: Hydrophobic inner/outer cutoffs.
        hbond_good: H-bond inner cutoff.
        core_cutoff: Euclidean interaction cutoff (Angstroms). Pairs with
            ``eucl > core_cutoff`` are skipped, so a Verlet list built with a
            skin margin scores the exact same set as a fresh ``core_cutoff``
            query (``query_ball_point`` is inclusive, matched by ``> core_cutoff``).
        prot_weight: Per-protein-atom occupancy weight (n_prot,); every term and
            gradient contribution from protein atom ``b`` is scaled by it (1.0 for
            ordinary atoms, fractional for altLoc conformers).
        inv_divisor: ``1 / (1 + w_rot * n_rot)``; applied to the gradient only
            (raw sums stay unweighted, matching the numpy path).

    Returns:
        ``(gauss1_raw, repulsion_raw, hydrophobic_raw, hbond_raw, atom_grad)``
        where ``atom_grad`` has shape (n_lig, 3).
    """
    n_lig = coords.shape[0]
    grad = np.zeros((n_lig, 3))
    g1_raw = 0.0
    rep_raw = 0.0
    hydro_raw = 0.0
    hb_raw = 0.0
    range_hydro = hydro_bad - hydro_good
    inv_hb = 1.0 / (-hbond_good)

    for p in range(i_idx.shape[0]):
        a = i_idx[p]
        b = j_idx[p]
        w = prot_weight[b]
        dx = coords[a, 0] - protein_coords[b, 0]
        dy = coords[a, 1] - protein_coords[b, 1]
        dz = coords[a, 2] - protein_coords[b, 2]
        eucl = np.sqrt(dx * dx + dy * dy + dz * dz)
        # Verlet gate: skip pairs beyond the core cutoff so a skin-padded list
        # scores identically to a fresh core-cutoff neighbor query.
        if eucl > core_cutoff:
            continue
        d = eucl - lig_radii[a] - prot_radii[b]

        z = (d - gauss1_offset) / gauss1_width
        g1 = np.exp(-z * z)
        g1_raw += g1 * w
        dfdd = w_gauss1 * (-2.0 * z / gauss1_width) * g1

        if d < 0.0:
            rep_raw += d * d * w
            dfdd += w_repulsion * 2.0 * d

        if lig_hydro[a] and prot_hydro[b]:
            if d <= hydro_good:
                hydro_raw += w
            elif d < hydro_bad:
                hydro_raw += (hydro_bad - d) / range_hydro * w
                dfdd += w_hydrophobic * (-1.0 / range_hydro)

        if (lig_don[a] and prot_acc[b]) or (lig_acc[a] and prot_don[b]):
            if d <= hbond_good:
                hb_raw += w
            elif d < 0.0:
                hb_raw += -d * inv_hb * w
                dfdd += w_hbond * (-inv_hb)

        if eucl > 1e-8:
            f = dfdd * w * inv_divisor / eucl
            grad[a, 0] += f * dx
            grad[a, 1] += f * dy
            grad[a, 2] += f * dz

    return g1_raw, rep_raw, hydro_raw, hb_raw, grad
