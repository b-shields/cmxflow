# I/O

Source and sink blocks handle reading and writing molecules in various file formats.

## Sources

::: cmxflow.sources.reader.MoleculeSourceBlock

### Supported Formats

| Extension | Format |
|-----------|--------|
| `.sdf` | SD file |
| `.sdf.gz` | Gzipped SD file |
| `.mol2` | Mol2 file |
| `.smi` | SMILES file |
| `.smi.gz` | Gzipped SMILES file |
| `.csv` | CSV with SMILES column |
| `.parquet` | Parquet with SMILES column |

### Reader Function

::: cmxflow.sources.reader.read_molecules

## Sinks

::: cmxflow.sinks.writer.MoleculeSinkBlock

### Writer Function

::: cmxflow.sinks.writer.write_molecules
