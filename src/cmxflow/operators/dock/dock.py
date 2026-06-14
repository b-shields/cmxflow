"""Molecular docking block for cmxflow workflows.

This module provides a MoleculeBlock implementation for docking ligands
into protein binding sites using the empirical (default Vinardo) scoring
function and rigid-body + torsional pose optimization.
"""

import dataclasses
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree

from cmxflow.operators.base import MoleculeBlock
from cmxflow.operators.dock.ec import (
    compute_gasteiger_charges,
    electrostatic_complementarity,
)
from cmxflow.operators.dock.pose import (
    OptimizationResult,
    PoseParams,
    optimize_dg_restarts,
    optimize_pose_cached,
)
from cmxflow.operators.dock.scaffold_index import (
    ScaffoldPoseStore,
    scaffold_key,
    scaffold_pose,
)
from cmxflow.operators.dock.score import (
    AtomTyping,
    EmpiricalParams,
    build_protein_tree,
    empirical_score_cached,
    get_atom_typing,
)
from cmxflow.operators.dock.template import template_dock
from cmxflow.parameter import (
    Categorical,
    Continuous,
    Integer,
)
from cmxflow.sources.reader import read_molecules

logger = logging.getLogger(__name__)

# Scaffold pose index (template docking). The cache lives at a fixed, conventional
# location in the execution directory so it is auto-discovered and reused across
# runs; only the on/off flag is propagated to workers (the path is derived from cwd).
_INDEX_DIR = Path(".cmxflow")
_INDEX_DB_NAME = "scaffold_index.db"
# Default flat-bottom core restraint for template docking: the scaffold may shift
# freely within _INDEX_CONSTRAINT_TOL Å (to relieve a substituent clash), then is
# resisted by a moderate spring. Tunable; deliberately far softer than a hard pin.
_INDEX_CONSTRAINT_WEIGHT = 25.0
_INDEX_CONSTRAINT_TOL = 0.5


class MoleculeDockBlock(MoleculeBlock):
    """MoleculeBlock for docking ligands into protein binding sites.

    Performs pose optimization using an empirical scoring function
    with configurable parameters. Supports both rigid-body and flexible
    (torsional) docking. Electrostatic complementarity (EC) is evaluated once on
    the final pose and reported as ``docking_ec`` — a standalone score, never
    part of the search.

    Two modes, both requiring a ``receptor`` and a ``site_reference``:

    - **Free docking** (``index_poses=False``, default): multi-start search
      (``n_starts``) anchored on the site-reference centroid.
    - **Scaffold-indexed docking** (``index_poses=True``): the first molecule of
      each Bemis-Murcko scaffold is docked fully and its core pose cached; later
      siblings transfer that pose and run a single constrained local search — much
      faster for congeneric series, and series-consistent. The ``site_reference``
      ligand is seeded as the first scaffold entry, so its experimentally grounded
      pose is the preferred template.

    Required Inputs:
        - receptor (file): Path to receptor PDB file.
        - site_reference (file): Molecule file (.sdf, .mol2, etc.) whose
          heavy-atom centroid defines the binding site center. Sobol restart
          samples are anchored to this pocket, so molecules dock from a freshly
          generated conformer — no preceding alignment step is required. It is
          the reference template seeded into the scaffold index when
          ``index_poses=True``. Technically optional: omit only for MCS/overlay
          refinement workflows where the input pose is already in the binding
          site (the search then recenters on the input conformer position).

    Output Properties:
        - docking_initial_pose_score: Score before optimization.
        - docking_score: Final optimized score (empirical + EC adjustment, plus
          ligand strain when ``score_strain=True``).
        - docking_empirical: Pure empirical score (without EC term).
        - docking_ec: Electrostatic complementarity of the final pose, in
          [-1, 1] (0.0 only when EC protein data is unavailable).
        - docking_strain: Ligand strain penalty — intramolecular energy added vs
          the input conformer (>=0). Reported regardless of ``score_strain``.
        - docking_converged: Whether optimization converged.
        When ``score_components=True`` (default), also writes:
        - docking_gauss1: Gaussian term contribution to docking_score.
        - docking_repulsion: Repulsion term contribution to docking_score.
        - docking_hydrophobic: Hydrophobic term contribution to docking_score.
        - docking_hbond: H-bond term contribution to docking_score.
        - docking_n_rot: Torsional entropy energetic term (w_rot * N_rot).
        - docking_scoring_function: Scoring weights used, for reproducibility.

    Example:
        ```python
        # site_reference recenters the search, so a fresh conformer docks
        # directly — no MoleculeAlignBlock required.
        workflow.add(
            MoleculeSourceBlock(),
            EnumerateStereoBlock(),
            ConformerGenerationBlock(),
            MoleculeDockBlock(
                receptor="protein.pdb",
                site_reference="crystal_ligand.sdf",
            ),
            MoleculeSinkBlock(),
        )
        ```

    Mutable Parameters:
        - w_gauss1: Vinardo Gaussian attractive term weight.
        - w_repulsion: Vinardo repulsion term weight.
        - w_hydrophobic: Vinardo hydrophobic term weight.
        - w_hbond: Vinardo hydrogen bond term weight.
        - w_rot: Torsional entropy divisor weight (0=pure Vinardo, 0.02=smina default).
        - n_starts: Number of L-BFGS-B restarts. 1 = local minimize from the
            input pose only. For blind docking (with site_reference), use
            1+2^k for ideal Sobol balance: 3, 5, 9, 17, 33, 65. Row 0 always
            minimizes from the aligned pose; rows 1+ sample the binding site box.
        - basin_hops: Iterated-local-search refinement steps per restart
            (0 = single minimize). Higher finds lower-energy poses at more cost.
        - max_iterations: Maximum L-BFGS-B iterations per restart.
        - box_size: Translation search box half-width in Angstroms (default 5.0).
            Centred on site_reference centroid when provided, otherwise on the
            input conformer position.
        - rigid: If True, only rigid-body optimization (no torsions).
        - index_poses: If True, scaffold-indexed (template) docking (see above).
            A mode toggle, not a search dimension — freeze it during optimization.
    """

    def __init__(
        self,
        score_components: bool = True,
        score_strain: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the molecular docking block.

        Args:
            score_components: If True (default), write per-term weighted score
                components as SDF properties on each docked molecule.
            score_strain: If True, add the ligand strain penalty (intramolecular
                energy added vs the input conformer, >=0) into ``docking_score``
                and into multistart selection. Default False keeps
                ``docking_score`` purely intermolecular (smina-comparable). The
                strain value is always written as ``docking_strain`` regardless.
            **kwargs: Passed to ``set_inputs``. Accepts the inputs ``receptor`` and
                ``site_reference`` (file paths) and any mutable parameter by name
                (``n_starts``, ``basin_hops``, ``max_iterations``, ``box_size``,
                ``rigid``, score weights, and ``index_poses``). ``index_poses`` is a
                bool (default ``False``): when ``True`` the block runs in
                scaffold-indexed (template) docking mode -- the first molecule of
                each Bemis-Murcko scaffold is docked fully and its scaffold pose
                cached at ``./.cmxflow/scaffold_index.db``; later molecules sharing
                that scaffold transfer the cached pose and run a single constrained
                local search (faster for congeneric series, and series-consistent).
                The cache persists across runs and is reused. Cache keys are
                namespaced by the docking parameters *and* the receptor/reference
                paths, so changing score weights or search settings, or pointing at
                a different target/site, never reuses a stale pose. ``index_poses``
                is a mode toggle: leave it out of the optimized parameter space.
        """
        super().__init__(
            name="MoleculeDock",
            input_files=["receptor", "site_reference"],
        )
        self._score_components = score_components
        self._score_strain = score_strain
        self._scaffold_store: ScaffoldPoseStore | None = None
        self._reference_seeded = False

        # Register mutable parameters
        self.mutable(
            # Vinardo score weights
            Continuous("w_gauss1", -0.045, -0.065, -0.025),
            Continuous("w_repulsion", 0.8, 0.8, 1.2),
            Continuous("w_hydrophobic", -0.035, -0.065, -0.015),
            Continuous("w_hbond", -0.6, -0.8, -0.4),
            Continuous("w_rot", 0.02, 0.0, 0.04),
            # Pose search. Per-mol runtime ~ n_starts x (1 + basin_hops) local
            # minima, each ~max_iterations L-BFGS-B steps; the bounds keep the
            # worst-case config tractable.
            #   n_starts hi=33: start diversity saturates near 32.
            Integer("n_starts", 32, 1, 33),
            #   basin_hops: extra iterated-local-search refinement per start.
            #   Default 0 (init + single minimize); hi=16 caps runtime.
            Integer("basin_hops", 0, 0, 16),
            #   max_iterations hi=200: L-BFGS-B converges well before then.
            Integer("max_iterations", 100, 50, 200),
            Continuous("box_size", 10.0, 5.0, 20.0),
            Categorical("rigid", False, [True, False]),
            # Initialization grid: max_distance_geometry_samples (M) ETKDGv3
            # conformers crossed with sobol_max_tries // M Sobol rigid placements;
            # the lowest-scoring starts at least diversity_rmsd apart are kept.
            Integer("sobol_max_tries", 2048, 512, 4096),
            Integer("max_distance_geometry_samples", 32, 1, 64),
            Continuous("diversity_rmsd", 1.0, 0.0, 5.0),
            # Mode toggle: scaffold-indexed (template) docking on/off
            Categorical("index_poses", False, [True, False]),
        )
        self.set_inputs(**kwargs)

        # Lazy-loaded protein scoring components
        self._protein_coords: np.ndarray | None = None
        self._protein_typing: AtomTyping | None = None
        self._protein_tree: cKDTree | None = None
        self._protein_ec_coords: np.ndarray | None = None
        self._protein_ec_charges: np.ndarray | None = None

    def _has_3d_conformer(self, mol: Chem.Mol) -> bool:
        """Check if molecule has at least one 3D conformer.

        Args:
            mol: RDKit Mol object to check.

        Returns:
            True if molecule has at least one 3D conformer, False otherwise.
        """
        if mol.GetNumConformers() == 0:
            return False
        return bool(mol.GetConformer().Is3D())

    def _load_site_reference(self) -> np.ndarray | None:
        """Compute binding site centroid from the site_reference reference file.

        Reads the first molecule with 3D coordinates from the ``site_reference``
        input file and returns its heavy-atom centroid. Supports .sdf, .mol2,
        and any other format handled by ``read_molecules``.

        Returns:
            (3,) centroid array, or None if no site_reference file is configured.

        Raises:
            FileNotFoundError: If the site_reference file does not exist.
            ValueError: If no molecule with 3D coordinates is found in the file.
        """
        site_path = self.input_files.get("site_reference")
        if site_path is None or str(site_path) == ".":
            return None

        site_path = Path(site_path)
        if not site_path.exists():
            raise FileNotFoundError(f"site_reference file does not exist: {site_path}")

        for mol in read_molecules(site_path, wrap=False):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            heavy = Chem.RemoveAllHs(mol)
            coords = np.array(heavy.GetConformer().GetPositions())
            centroid = np.mean(coords, axis=0)
            logger.debug(
                "site_reference loaded from %s: centroid=(%.2f, %.2f, %.2f)",
                site_path.name,
                *centroid,
            )
            return centroid

        raise ValueError(
            f"No molecule with 3D coordinates found in site_reference file: {site_path}"
        )

    def _index_rel_path(self, key: str) -> str:
        """Resolve an input file path relative to the cache dir (``.cmxflow/``).

        Used to namespace cached poses by target/site so one shared cache can serve
        several receptors or reference sites without collisions. Returns ``""`` when
        the input is unset.
        """
        path = self.input_files.get(key)
        if path is None or str(path) == ".":
            return ""
        return os.path.relpath(os.path.abspath(str(path)), os.path.abspath(_INDEX_DIR))

    def _index_namespace(self) -> str:
        """Hash of the docking params + target/site that define a cached pose.

        Namespaces the scaffold cache so changing score weights or search settings
        (e.g. across parameter-optimization trials), or pointing at a different
        receptor/reference, never reuses a stale template.
        """
        items = sorted((name, p.get()) for name, p in self.params.items())
        payload = (
            f"{items}|score_strain={self._score_strain}"
            f"|receptor={self._index_rel_path('receptor')}"
            f"|site_reference={self._index_rel_path('site_reference')}"
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _indexing_enabled(self) -> bool:
        """Whether scaffold-indexed (template) docking is active."""
        return bool(self.get_param("index_poses"))

    def _ensure_index_ready(self) -> ScaffoldPoseStore:
        """Open the per-process store and seed the reference scaffold once."""
        if self._scaffold_store is None:
            self._scaffold_store = ScaffoldPoseStore(_INDEX_DIR / _INDEX_DB_NAME)
        if not self._reference_seeded:
            self._reference_seeded = True
            self._seed_reference_scaffold()
        return self._scaffold_store

    def _seed_reference_scaffold(self) -> None:
        """Seed the ``site_reference`` ligand's scaffold pose (idempotent).

        The reference is the experimentally grounded pose, so it becomes the
        preferred template for the series core and is seeded deterministically.
        """
        assert self._scaffold_store is not None
        site_path = self.input_files.get("site_reference")
        if site_path is None or str(site_path) == ".":
            return
        site_path = Path(site_path)
        if not site_path.exists():
            return
        for ref in read_molecules(site_path, wrap=False):
            if ref is None or ref.GetNumConformers() == 0:
                continue
            key = scaffold_key(ref)
            posed = scaffold_pose(ref)
            if key is not None and posed is not None:
                self._scaffold_store.put(f"{self._index_namespace()}:{key}", posed)
            return

    def _load_receptor(self) -> None:
        """Load and validate receptor from a PDB file.

        Raises:
            FileNotFoundError: If the receptor PDB file does not exist.
            ValueError: If no valid reference molecules with 3D conformers found.
        """
        receptor_path = Path(self.input_files["receptor"])
        if not receptor_path.exists():
            raise FileNotFoundError(f"Receptor path does not exist: {receptor_path}.")

        mol = Chem.MolFromPDBFile(str(receptor_path))

        # Prepare receptor with explicit H for EC scoring
        mol_with_h = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol_with_h)
        self._protein_ec_coords = np.array(mol_with_h.GetConformer().GetPositions())
        self._protein_ec_charges = compute_gasteiger_charges(mol_with_h)

        # United atom for empirical scoring
        mol = Chem.RemoveAllHs(mol)

        if not self._has_3d_conformer(mol):
            raise ValueError(f"Receptor {receptor_path} does not have a 3D conformer.")

        protein_conf = mol.GetConformer()
        self._protein_coords = np.array(protein_conf.GetPositions())
        self._protein_typing = get_atom_typing(mol)
        self._protein_tree = build_protein_tree(self._protein_coords)

    def _prune_to_single_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Reduce molecule to single conformer for docking.

        Args:
            mol: RDKit Mol object, possibly with multiple conformers.

        Returns:
            Molecule with at most one conformer.
        """
        num_confs = mol.GetNumConformers()
        if num_confs <= 1:
            return mol
        else:
            logger.info(
                f"Molecule has {num_confs} conformers. Docking only conformer 0."
            )
            conf = mol.GetConformer()
            new_mol = Chem.Mol(mol)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(Chem.Conformer(conf), assignId=True)
            return new_mol

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Dock a ligand molecule into the receptor binding site.

        Args:
            mol: Ligand RDKit Mol with 3D conformer.

        Returns:
            Docked molecule with optimized pose and docking properties,
            or None if docking fails or input lacks 3D conformers.
        """
        if not self._has_3d_conformer(mol):
            logger.debug(
                "Input molecule does not have 3D conformers. Skipping docking."
            )
            return None

        mol = self._prune_to_single_conformer(mol)

        if self._protein_coords is None:
            self._load_receptor()
        assert isinstance(self._protein_coords, np.ndarray)
        assert isinstance(self._protein_typing, AtomTyping)

        score_params = EmpiricalParams(
            w_gauss1=self.get_param("w_gauss1"),
            w_repulsion=self.get_param("w_repulsion"),
            w_hydrophobic=self.get_param("w_hydrophobic"),
            w_hbond=self.get_param("w_hbond"),
            w_rot=self.get_param("w_rot"),
        )
        box_size = self.get_param("box_size")
        rigid_only = self.get_param("rigid")
        site_center = self._load_site_reference()

        # Scaffold-indexed (template) docking: on a cache hit, transfer the stored
        # scaffold pose and run a single constrained local search instead of a full
        # multistart search. Cache keys are namespaced by the docking parameters.
        index_key: str | None = None
        store: ScaffoldPoseStore | None = None
        if self._indexing_enabled():
            store = self._ensure_index_ready()
            key = scaffold_key(mol)
            if key is not None:
                index_key = f"{self._index_namespace()}:{key}"

        result: OptimizationResult | None = None
        indexed = False
        n_starts_used = 0

        if index_key is not None and store is not None:
            core = store.get(index_key)
            if core is not None:
                result = template_dock(
                    mol,
                    core,
                    self._protein_coords,
                    self._protein_typing,
                    constraint_weight=_INDEX_CONSTRAINT_WEIGHT,
                    constraint_tol=_INDEX_CONSTRAINT_TOL,
                    score_params=score_params,
                    max_iterations=self.get_param("max_iterations"),
                    basin_hops=self.get_param("basin_hops"),
                    optimize_torsions=not rigid_only,
                    translation_bounds=(-box_size, box_size),
                    protein_tree=self._protein_tree,
                )
                if result is not None:
                    indexed = True
                    n_starts_used = 1

        if result is None:
            # Full search. Phase 1: distance-geometry initialization — score sampled
            # poses (a DG conformer ensemble crossed with a Sobol rigid grid) to find
            # good basins.
            init_params = PoseParams(
                translation_bounds=(-box_size, box_size),
                n_starts=self.get_param("n_starts"),
            )
            starts = optimize_dg_restarts(
                mol,
                protein_coords=self._protein_coords,
                protein_typing=self._protein_typing,
                params=init_params,
                score_params=score_params,
                site_center=site_center,
                rigid=rigid_only,
                max_tries=self.get_param("sobol_max_tries"),
                max_distance_geometry_samples=self.get_param(
                    "max_distance_geometry_samples"
                ),
                diversity_rmsd=self.get_param("diversity_rmsd"),
                protein_tree=self._protein_tree,
            )

            # Phase 2: L-BFGS-B refinement from each starting pose.
            refine_params = PoseParams(
                max_iterations=self.get_param("max_iterations"),
                translation_bounds=(-box_size, box_size),
                optimize_torsions=not rigid_only,
                n_starts=1,
                basin_hops=self.get_param("basin_hops"),
            )

            # Selection objective: intermolecular score, plus strain when the strain
            # toggle is on (so multistart picks the pose we will actually report).
            def _effective(r: OptimizationResult) -> float:
                return r.score + (r.strain if self._score_strain else 0.0)

            for idx, (_, start_mol) in enumerate(starts):
                candidate = optimize_pose_cached(
                    start_mol,
                    protein_coords=self._protein_coords,
                    protein_typing=self._protein_typing,
                    # Distinct seed per chain so basin-hopping walks decorrelate.
                    params=dataclasses.replace(refine_params, seed=idx),
                    score_params=score_params,
                    site_center=None,
                    protein_tree=self._protein_tree,
                )
                if result is None or _effective(candidate) < _effective(result):
                    result = candidate

            assert result is not None
            result = dataclasses.replace(result, initial_score=starts[0][0])
            n_starts_used = len(starts)

            # Cache this scaffold's pose for later siblings (first-writer-wins).
            if index_key is not None and store is not None:
                posed = scaffold_pose(result.mol)
                if posed is not None:
                    store.put(index_key, posed)

        # Electrostatic complementarity: a reporting-only score evaluated once on
        # the final pose (never part of the search). Protein charges were computed
        # once at receptor load. Reported so it can be selected as a ranking score.
        docking_ec = 0.0
        if self._protein_ec_coords is not None and self._protein_ec_charges is not None:
            docking_ec = electrostatic_complementarity(
                result.mol, self._protein_ec_coords, self._protein_ec_charges
            )

        # Set properties. docking_score optionally includes the ligand strain
        # penalty; docking_empirical stays pure intermolecular (smina-comparable).
        docking_score = result.score + (result.strain if self._score_strain else 0.0)
        result.mol.SetIntProp("docking_n_starts_used", n_starts_used)
        result.mol.SetBoolProp("docking_indexed", indexed)
        result.mol.SetDoubleProp("docking_initial_pose_score", result.initial_score)
        result.mol.SetDoubleProp("docking_score", docking_score)
        result.mol.SetDoubleProp("docking_empirical", result.score)
        result.mol.SetDoubleProp("docking_strain", result.strain)
        result.mol.SetDoubleProp("docking_ec", docking_ec)
        result.mol.SetBoolProp("docking_converged", result.converged)

        # Per-term score components (reporting path)
        if self._score_components:
            assert isinstance(self._protein_coords, np.ndarray)
            assert isinstance(self._protein_typing, AtomTyping)
            ligand_heavy = Chem.RemoveAllHs(result.mol)
            comps = empirical_score_cached(
                ligand_heavy,
                self._protein_coords,
                self._protein_typing,
                params=score_params,
                protein_tree=self._protein_tree,
            )
            result.mol.SetDoubleProp("docking_gauss1", comps.gauss1)
            result.mol.SetDoubleProp("docking_repulsion", comps.repulsion)
            result.mol.SetDoubleProp("docking_hydrophobic", comps.hydrophobic)
            result.mol.SetDoubleProp("docking_hbond", comps.hbond)
            result.mol.SetDoubleProp("docking_n_rot", comps.n_rot * comps.w_rot)
            result.mol.SetProp(
                "docking_scoring_function",
                f"(gauss1) {score_params.w_gauss1:.3f} "
                f"(repulsion) {score_params.w_repulsion:.3f} "
                f"(hydrophobic) {score_params.w_hydrophobic:.3f} "
                f"(hbond) {score_params.w_hbond:.3f} "
                f"(w_rot) {score_params.w_rot:.3f}",
            )

        return result.mol

    def check_output(self, arg: Any) -> bool:
        """Validate that output molecule has docking properties.

        Args:
            arg: Output molecule to validate.

        Returns:
            True if the molecule has valid docking output, False otherwise.
        """
        if not isinstance(arg, Chem.Mol):
            logger.info(f"Docking failed: output is of type {type(arg)}")
            return False

        if arg.GetNumConformers() == 0:
            logger.info("Docking failed: output molecule has no conformers")
            return False

        required = (
            "docking_initial_pose_score",
            "docking_score",
            "docking_ec",
            "docking_converged",
        )
        for key in required:
            if not arg.HasProp(key):
                logger.info(f"Docking failed: missing {key} property")
                return False

        return True
