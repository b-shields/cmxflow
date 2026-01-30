"""Tests for gzip support in molecule readers."""

import gzip
import tempfile
from pathlib import Path

import pytest

from cmxflow.sources import (
    read_mol2,
    read_molecules,
    read_smi_gz,
)
from cmxflow.sources.reader import _parse_suffix


class TestParseSuffix:
    """Tests for _parse_suffix helper function."""

    def test_plain_suffix(self) -> None:
        """Test parsing plain file suffixes."""
        assert _parse_suffix(Path("file.sdf")) == (".sdf", False)
        assert _parse_suffix(Path("file.mol2")) == (".mol2", False)
        assert _parse_suffix(Path("file.smi")) == (".smi", False)
        assert _parse_suffix(Path("file.csv")) == (".csv", False)
        assert _parse_suffix(Path("file.parquet")) == (".parquet", False)

    def test_gzipped_suffix(self) -> None:
        """Test parsing gzipped file suffixes."""
        assert _parse_suffix(Path("file.sdf.gz")) == (".sdf", True)
        assert _parse_suffix(Path("file.smi.gz")) == (".smi", True)
        assert _parse_suffix(Path("file.csv.gz")) == (".csv", True)
        assert _parse_suffix(Path("file.parquet.gz")) == (".parquet", True)

    def test_case_insensitive(self) -> None:
        """Test that suffix parsing is case insensitive."""
        assert _parse_suffix(Path("file.SDF")) == (".sdf", False)
        assert _parse_suffix(Path("file.SDF.GZ")) == (".sdf", True)

    def test_no_suffix(self) -> None:
        """Test parsing file with no suffix."""
        assert _parse_suffix(Path("file")) == ("", False)

    def test_path_with_dots(self) -> None:
        """Test parsing path with dots in directory names."""
        assert _parse_suffix(Path("/path/to/v1.0/file.sdf")) == (".sdf", False)
        assert _parse_suffix(Path("/path/to/v1.0/file.sdf.gz")) == (".sdf", True)


class TestReadSmiGz:
    """Tests for read_smi_gz function."""

    def test_read_smi_gz(self) -> None:
        """Test reading a gzipped SMILES file."""
        with tempfile.NamedTemporaryFile(suffix=".smi.gz", delete=False) as f:
            path = Path(f.name)

        try:
            with gzip.open(path, "wt") as f:
                f.write("CCO ethanol\n")
                f.write("CC propane\n")
                f.write("C methane\n")

            mols = list(read_smi_gz(path))
            assert len(mols) == 3
            assert mols[0].GetProp("_Name") == "ethanol"
            assert mols[1].GetProp("_Name") == "propane"
            assert mols[2].GetProp("_Name") == "methane"
        finally:
            path.unlink()

    def test_read_smi_gz_via_dispatcher(self) -> None:
        """Test reading gzipped SMILES via read_molecules dispatcher."""
        with tempfile.NamedTemporaryFile(suffix=".smi.gz", delete=False) as f:
            path = Path(f.name)

        try:
            with gzip.open(path, "wt") as f:
                f.write("CCO ethanol\n")
                f.write("CC propane\n")

            mols = list(read_molecules(path))
            assert len(mols) == 2
        finally:
            path.unlink()


class TestReadCsvGz:
    """Tests for reading gzipped CSV files."""

    def test_read_csv_gz_via_dispatcher(self) -> None:
        """Test that pandas handles gzipped CSV automatically."""
        with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as f:
            path = Path(f.name)

        try:
            with gzip.open(path, "wt") as f:
                f.write("SMILES,Name\n")
                f.write("CCO,ethanol\n")
                f.write("CC,propane\n")

            mols = list(read_molecules(path))
            assert len(mols) == 2
        finally:
            path.unlink()


class TestParquetGzError:
    """Tests for parquet.gz rejection."""

    def test_parquet_gz_raises_error(self) -> None:
        """Test that .parquet.gz raises a clear error."""
        path = Path("/fake/path/file.parquet.gz")
        with pytest.raises(ValueError, match="Gzipped parquet not supported"):
            # We don't need the file to exist since the error is raised before reading
            list(read_molecules(path))


class TestPlainFormatsStillWork:
    """Tests to ensure plain (non-gzipped) formats still work."""

    def test_read_mol2_plain(self) -> None:
        """Test that plain mol2 reading still works."""
        mol2_content = """@<TRIPOS>MOLECULE
ethanol
 9 8 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1         0.0000    0.0000    0.0000 C.3       1 LIG1       0.0000
      2 C2         1.5000    0.0000    0.0000 C.3       1 LIG1       0.0000
      3 O1         2.0000    1.2000    0.0000 O.3       1 LIG1       0.0000
      4 H1        -0.3500   -0.5000    0.9000 H         1 LIG1       0.0000
      5 H2        -0.3500   -0.5000   -0.9000 H         1 LIG1       0.0000
      6 H3        -0.3500    1.0000    0.0000 H         1 LIG1       0.0000
      7 H4         1.8500   -0.5000   -0.9000 H         1 LIG1       0.0000
      8 H5         1.8500   -0.5000    0.9000 H         1 LIG1       0.0000
      9 H6         2.9500    1.2000    0.0000 H         1 LIG1       0.0000
@<TRIPOS>BOND
     1     1     2 1
     2     2     3 1
     3     1     4 1
     4     1     5 1
     5     1     6 1
     6     2     7 1
     7     2     8 1
     8     3     9 1
"""
        with tempfile.NamedTemporaryFile(suffix=".mol2", delete=False, mode="w") as f:
            f.write(mol2_content)
            path = Path(f.name)

        try:
            mols = list(read_mol2(path))
            assert len(mols) == 1
            assert mols[0].GetNumAtoms() > 0
        finally:
            path.unlink()
