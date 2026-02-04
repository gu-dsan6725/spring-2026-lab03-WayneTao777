# Workflow: run-wine-eda

## Goal
Run exploratory data analysis on the UCI Wine dataset and save plots/artifacts.

## Steps
1. Load dataset with `load_wine()` and convert to polars DataFrame.
2. Compute summary statistics and missing value counts.
3. Plot distributions and correlation heatmap.
4. Check class balance and save a class balance plot.
5. Identify outliers via IQR and log counts.
6. Save all artifacts to `output/antigravity/`.

## Quality Checks
- Confirm class balance and no missing values.
- Ensure plots and JSON files are written.
