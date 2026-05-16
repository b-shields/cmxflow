# Contributing

Contributions are welcome! This is a side project, so reviews may take some time, but PRs are appreciated.

## Before Submitting

1. **Open an issue first** for significant changes to discuss the approach
2. **Fork the repo** and create a feature branch from `main`
3. **Follow the code style** — run `mypy`, `black`, and `ruff` before committing (or install provided precommit hooks)

## PR Requirements

- **Clear description** of the bug fixed or feature added
- **Minimal reproducible example** demonstrating the change
- **Tests** covering new functionality (`pytest`)
- **Type hints** for all new code
- **Docstrings** following Google conventions

## Development Setup

```bash
conda config --set solver libmamba
conda env create -f conda.yml
conda activate cmxflow
poetry install
pre-commit install  # Ensures formatting/linting on commit
```

## Running Tests

```bash
pytest tests/
```
