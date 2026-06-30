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


@njit(cache=True, fastmath=True)
def score_pocket(
    coords: np.ndarray,
    pocket_coords: np.ndarray,
    lig_radii: np.ndarray,
    pocket_radii: np.ndarray,
    lig_hydro: np.ndarray,
    pocket_hydro: np.ndarray,
    lig_don: np.ndarray,
    lig_acc: np.ndarray,
    pocket_don: np.ndarray,
    pocket_acc: np.ndarray,
    pocket_weight: np.ndarray,
    gauss1_offset: float,
    gauss1_width: float,
    hydro_good: float,
    hydro_bad: float,
    hbond_good: float,
    core_cutoff: float,
) -> tuple[float, float, float, float]:
    """Vinardo term sums over a fixed pocket subset -- score only, no gradient.

    Initialization-screening hot path. Every candidate placement of a molecule
    sits within a bounded radius of the binding-site anchor, so the caller builds
    one protein "pocket" subset (atoms that could contact *any* placement) once
    and scores every candidate against it here with no per-call KD-tree query.
    A plain double loop over ``n_lig x n_pocket`` with the same ``eucl >
    core_cutoff`` gate as :func:`score_grad_pairs` -- so the term sums are
    identical to the per-call path, minus the gradient scatter that screening
    does not need.

    Args:
        coords: Ligand heavy-atom coordinates (n_lig, 3).
        pocket_coords: Pocket protein coordinates (n_pocket, 3).
        lig_radii, pocket_radii: Per-atom Vinardo radii.
        lig_hydro, pocket_hydro: Per-atom hydrophobic masks.
        lig_don, lig_acc, pocket_don, pocket_acc: H-bond donor/acceptor masks.
        pocket_weight: Per-pocket-atom occupancy weight (n_pocket,).
        gauss1_offset, gauss1_width: Gaussian center/width.
        hydro_good, hydro_bad: Hydrophobic inner/outer cutoffs.
        hbond_good: H-bond inner cutoff.
        core_cutoff: Euclidean interaction cutoff (Angstroms).

    Returns:
        ``(gauss1_raw, repulsion_raw, hydrophobic_raw, hbond_raw)`` -- the four
        *unweighted* term sums (caller applies weights + torsion divisor).
    """
    n_lig = coords.shape[0]
    n_pocket = pocket_coords.shape[0]
    g1_raw = 0.0
    rep_raw = 0.0
    hydro_raw = 0.0
    hb_raw = 0.0
    range_hydro = hydro_bad - hydro_good
    inv_hb = 1.0 / (-hbond_good)

    for a in range(n_lig):
        ax = coords[a, 0]
        ay = coords[a, 1]
        az = coords[a, 2]
        ra = lig_radii[a]
        ha = lig_hydro[a]
        da = lig_don[a]
        aa = lig_acc[a]
        for b in range(n_pocket):
            dx = ax - pocket_coords[b, 0]
            dy = ay - pocket_coords[b, 1]
            dz = az - pocket_coords[b, 2]
            eucl = np.sqrt(dx * dx + dy * dy + dz * dz)
            if eucl > core_cutoff:
                continue
            w = pocket_weight[b]
            d = eucl - ra - pocket_radii[b]

            z = (d - gauss1_offset) / gauss1_width
            g1_raw += np.exp(-z * z) * w

            if d < 0.0:
                rep_raw += d * d * w

            if ha and pocket_hydro[b]:
                if d <= hydro_good:
                    hydro_raw += w
                elif d < hydro_bad:
                    hydro_raw += (hydro_bad - d) / range_hydro * w

            if (da and pocket_acc[b]) or (aa and pocket_don[b]):
                if d <= hbond_good:
                    hb_raw += w
                elif d < 0.0:
                    hb_raw += -d * inv_hb * w

    return g1_raw, rep_raw, hydro_raw, hb_raw


@njit(cache=True, fastmath=True)
def score_pocket_batch(
    coords_batch: np.ndarray,
    pocket_coords: np.ndarray,
    lig_radii: np.ndarray,
    pocket_radii: np.ndarray,
    lig_hydro: np.ndarray,
    pocket_hydro: np.ndarray,
    lig_don: np.ndarray,
    lig_acc: np.ndarray,
    pocket_don: np.ndarray,
    pocket_acc: np.ndarray,
    pocket_weight: np.ndarray,
    gauss1_offset: float,
    gauss1_width: float,
    hydro_good: float,
    hydro_bad: float,
    hbond_good: float,
    core_cutoff: float,
) -> np.ndarray:
    """Batched :func:`score_pocket`: term sums for K candidate poses at once.

    Same per-pose math as :func:`score_pocket`, looped over the leading
    candidate axis inside numba so the restart screen pays one kernel dispatch
    for the whole grid instead of one per candidate. All candidates of a molecule
    share the ligand typing and pocket subset, which are passed once.

    Args:
        coords_batch: Candidate ligand coordinates (K, n_lig, 3).
        pocket_coords: Pocket protein coordinates (n_pocket, 3).
        lig_radii, pocket_radii: Per-atom Vinardo radii.
        lig_hydro, pocket_hydro: Per-atom hydrophobic masks.
        lig_don, lig_acc, pocket_don, pocket_acc: H-bond donor/acceptor masks.
        pocket_weight: Per-pocket-atom occupancy weight (n_pocket,).
        gauss1_offset, gauss1_width: Gaussian center/width.
        hydro_good, hydro_bad: Hydrophobic inner/outer cutoffs.
        hbond_good: H-bond inner cutoff.
        core_cutoff: Euclidean interaction cutoff (Angstroms).

    Returns:
        ``(K, 4)`` array of unweighted ``(gauss1, repulsion, hydrophobic, hbond)``
        term sums per candidate (caller applies weights + torsion divisor).
    """
    n_cand = coords_batch.shape[0]
    n_lig = coords_batch.shape[1]
    n_pocket = pocket_coords.shape[0]
    out = np.zeros((n_cand, 4))
    range_hydro = hydro_bad - hydro_good
    inv_hb = 1.0 / (-hbond_good)

    for k in range(n_cand):
        g1_raw = 0.0
        rep_raw = 0.0
        hydro_raw = 0.0
        hb_raw = 0.0
        for a in range(n_lig):
            ax = coords_batch[k, a, 0]
            ay = coords_batch[k, a, 1]
            az = coords_batch[k, a, 2]
            ra = lig_radii[a]
            ha = lig_hydro[a]
            da = lig_don[a]
            aa = lig_acc[a]
            for b in range(n_pocket):
                dx = ax - pocket_coords[b, 0]
                dy = ay - pocket_coords[b, 1]
                dz = az - pocket_coords[b, 2]
                eucl = np.sqrt(dx * dx + dy * dy + dz * dz)
                if eucl > core_cutoff:
                    continue
                w = pocket_weight[b]
                d = eucl - ra - pocket_radii[b]

                z = (d - gauss1_offset) / gauss1_width
                g1_raw += np.exp(-z * z) * w

                if d < 0.0:
                    rep_raw += d * d * w

                if ha and pocket_hydro[b]:
                    if d <= hydro_good:
                        hydro_raw += w
                    elif d < hydro_bad:
                        hydro_raw += (hydro_bad - d) / range_hydro * w

                if (da and pocket_acc[b]) or (aa and pocket_don[b]):
                    if d <= hbond_good:
                        hb_raw += w
                    elif d < 0.0:
                        hb_raw += -d * inv_hb * w
        out[k, 0] = g1_raw
        out[k, 1] = rep_raw
        out[k, 2] = hydro_raw
        out[k, 3] = hb_raw
    return out
