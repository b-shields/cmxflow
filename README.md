# cmxflow

Automated cheminformatics workflow optimization.

## Installation

```bash
conda env create -f conda.yml
conda activate cmxflow
poetry install
pre-commit install
```

## To Do

- The score blocks need to be trainable but then just compute scores and let mols pass through if they are followed by a SinkBlock.
