# Code Style Guide

## Python Standards

- Follow PEP 8 with a line length of 100 characters
- Use type annotations for all function parameters
- Place one parameter per line for readability
- Use two blank lines between function and class definitions
- Private functions start with underscore and go at the top of the file

## Data Processing

- Use polars instead of pandas for all data manipulation
- Use scikit-learn for ML pipelines and evaluation
- Use XGBoost for gradient boosting models

## Logging

Use this exact logging configuration in every Python file:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
```

## Quality Checks

After writing any Python file, run:
1. `uv run ruff check --fix <filename>`
2. `uv run ruff format <filename>`
3. `uv run python -m py_compile <filename>`
