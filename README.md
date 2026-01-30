# cmxflow

Automated cheminformatics workflow optimization.

## Installation

```bash
conda env create -f conda.yml
conda activate cmxflow
poetry install
pre-commit install
```

## Project structure

```
cmxflow/
├── src/
│   └── cmxflow/
│       ├── __init__.py
│       ├── sources/           # Input readers
│       │   ├── __init__.py
│       │   ├── reader.py      # Main reader with format parsed by file extension
│       │   ├── table.py       # Reading tabular files (.smi, .csv, .parquet)
│       │   └── molecule.py    # Reading molecule files (.sdf, .sdf.gz)
│       ├── sinks/             # Output writers
│       │   ├── __init__.py
│       │   ├── writer.py      # Main file writer with format parsed by file extension
│       │   ├── table.py       # Writing tabular files (.smi, .csv, .parquet)
│       │   └── molecule.py    # Writing molecule files (.sdf, .sdf.gz)
│       ├── operators/         # Single molecule operations
│       │   ├── __init__.py
│       │   ├── base.py        # Operator base class inheriting from block base
│       │   └── sim2d.py       # 2D fingerprint similarity
│       ├── objectives/        # Optimization objectives
│       │   ├── __init__.py
│       │   ├── base.py        # Objective base class
│       │   └── enrichment.py  # Virtual screening objective (e.g., benchmark enrichment factor)
│       ├── optimizers/        # Objective optimizers
│       │   ├── __init__.py
│       │   ├── base.py        # Optimizer base class (optimize block mutable parameters in a workflow)
│       │   └── optuna.py      # An optuna optimizer
│       ├── parameter.py       # Defines scope of mutable parameters
│       ├── block.py           # Base class for molecule operation blocks
│       ├── workflow.py        # Base class for full workflows
│       └── cli.py             # Command-line interface built with rich
├── tests/
│   ├── __init__.py
│   ├── test_sources.py
│   ├── test_sinks.py
│   ├── test_operators.py
│   ├── test_objectives.py
│   ├── test_optimizers.py
│   └── fixtures/              # Test input files
└── examples/
    └── basic_usage.ipynb
```

Ideas:
- `block.Block` base class:
    - Defines mutable parameters and sets them at run time
    - Stores block input files to surface as required workflow inputs
    - Requires definition of `forward` method that is mapped to `__call__`
    - Optional definition of `__check__` method that checks for required input properties
    - Has a `__repr__` method that generates an a nice colored text view of the block
- `workflow.Workflow` class:
    - Allows for dynamic building of workflows from blocks
    - Must start with a source and end with a sink
    - Surfaces required input files from constituent blocks
    - Surfaces mutable parameters and allows them to be set across blocks
    - Has a `__repr__` method that generates a nice colored text view of how blocks are connected

Plan the implementation of `cmxflow.operators.align`. It should include a `MoleculeAlignBlock` class that inherits from `MoleculeBlock`. It should require a input_file called query with molecules (and compute 3D conformers only if not no present in the input) and compute align molecules in different ways (crippen, mmff, mcs and anything else in rdkit) controled by mutable parameters. If should align all conformers of an input molecule to all references and return only the conformer with the highest shape similarity (delete the others).
