"""Conformer generation and stereoisomer enumeration blocks."""

import logging
from collections.abc import Iterator
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from cmxflow.operators.base import MoleculeBlock
from cmxflow.parameter import Categorical, Continuous, Integer

logger = logging.getLogger(__name__)


class EnumerateStereoBlock(MoleculeBlock):
    """Block for enumerating stereoisomers of molecules.

    This is a 1:N transform that yields all possible stereoisomers for each
    input molecule. Properties from the input molecule are copied to each
    output stereoisomer.
    """

    def __init__(self) -> None:
        """Initialize the stereoisomer enumeration block."""
        super().__init__()

    def forward(self, mol: Chem.Mol) -> Iterator[Chem.Mol]:
        """Enumerate all stereoisomers of a molecule.

        Args:
            mol: Input RDKit Mol object.

        Yields:
            Stereoisomers of the input molecule with properties preserved.
        """
        opts = StereoEnumerationOptions(unique=True)
        isomers = EnumerateStereoisomers(mol, options=opts)

        # Get properties from input molecule
        props = mol.GetPropsAsDict()

        for isomer in isomers:
            # Copy properties to each stereoisomer
            for key, value in props.items():
                if isinstance(value, float):
                    isomer.SetDoubleProp(key, value)
                elif isinstance(value, int):
                    isomer.SetIntProp(key, value)
                else:
                    isomer.SetProp(key, str(value))
            yield isomer

    def __call__(self, iter: Iterator[Any]) -> Iterator[Chem.Mol]:
        """Execute the block on an iterator of molecules.

        Overrides the base __call__ to handle 1:N transformation where
        each input molecule can produce multiple stereoisomers.

        Args:
            iter: Iterator of input molecules to process.

        Yields:
            Stereoisomers that pass input and output checks.
        """
        for arg in iter:
            if not self.check_input(arg):
                continue
            for out in self.forward(arg):
                if self.check_output(out):
                    yield out


class ConformerGenerationBlock(MoleculeBlock):
    """Block for generating 3D conformers of molecules.

    Uses RDKit's ETKDGv3 algorithm to generate conformers. Molecules must
    have fully specified stereochemistry before conformer generation.

    Attributes:
        params: Dictionary of mutable parameters (numConfs, pruneRmsThresh,
            useRandomCoords).
    """

    def __init__(self) -> None:
        """Initialize the conformer generation block."""
        super().__init__()

        # Register mutable parameters
        self.mutable(
            Integer("numConfs", default=1, low=1, high=100),
            Continuous("pruneRmsThresh", default=0.5, low=0.0, high=5.0),
            Categorical("useRandomCoords", default=False, choices=[True, False]),
        )

    def check_input(self, arg: Any) -> bool:
        """Validate that input is an RDKit Mol with specified stereochemistry.

        Args:
            arg: Input item to validate.

        Returns:
            True if the input is valid, False otherwise.

        Raises:
            ValueError: If molecule has unspecified stereocenters.
        """
        if not super().check_input(arg):
            return False

        mol = arg
        # Check for unspecified chiral centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for _, stereo in chiral_centers:
            if stereo == "?":
                raise ValueError(
                    "Molecule has unspecified chiral centers. "
                    "Use EnumerateStereoBlock first to enumerate stereoisomers."
                )

        # Check for unspecified double bond stereochemistry
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                stereo = bond.GetStereo()
                if stereo == Chem.BondStereo.STEREOANY:
                    raise ValueError(
                        "Molecule has unspecified double bond stereochemistry. "
                        "Use EnumerateStereoBlock first to enumerate stereoisomers."
                    )

        return True

    def check_output(self, arg: Any) -> bool:
        """Validate that output molecule has 3D coordinates.

        Args:
            arg: Output molecule to validate.

        Returns:
            True if the molecule has 3D coordinates, False otherwise.
        """
        if not isinstance(arg, Chem.Mol):
            return False

        if arg.GetNumConformers() == 0:
            logger.info("Conformer generation failed: molecule has no conformers")
            return False

        conf = arg.GetConformer()
        if not conf.Is3D():
            logger.info("Conformer generation failed: coordinates are not 3D")
            return False

        return True

    def forward(self, mol: Chem.Mol) -> Chem.Mol:
        """Generate 3D conformers for a molecule.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            Molecule with 3D conformers added.
        """
        # Get parameters
        num_confs = self.params["numConfs"].get()
        prune_rms_thresh = self.params["pruneRmsThresh"].get()
        use_random_coords = self.params["useRandomCoords"].get()

        # Work on a copy to avoid modifying the input
        mol = Chem.Mol(mol)

        # Remove existing conformers
        mol.RemoveAllConformers()

        # Add hydrogens for conformer generation
        mol = Chem.AddHs(mol)

        # Set up ETKDGv3 parameters
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = prune_rms_thresh
        params.useRandomCoords = use_random_coords
        params.numThreads = 0  # Use all available threads

        # Generate conformers
        rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

        return mol
