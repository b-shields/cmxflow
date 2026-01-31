# cmxflow

Automated cheminformatics workflow optimization.

## Installation

Development install.
```bash
# Build base environment
conda config --set solver libmamba
conda env create -f conda.yml
conda activate cmxflow

# Install cmxflow
poetry install
pre-commit install
```
