# I/O

Source and sink blocks handle reading and writing molecules in various file formats.

## Supported Formats

| Extension | Format |
|-----------|--------|
| `.sdf` | SD file |
| `.sdf.gz` | Gzipped SD file |
| `.mol2` | Mol2 file |
| `.smi` | SMILES file |
| `.smi.gz` | Gzipped SMILES file |
| `.csv` | CSV with SMILES column |
| `.parquet` | Parquet with SMILES column |

## Sources

::: cmxflow.sources.reader.MoleculeSourceBlock
    options:
      members: false

## Sinks

::: cmxflow.sinks.writer.MoleculeSinkBlock
    options:
      members: false
