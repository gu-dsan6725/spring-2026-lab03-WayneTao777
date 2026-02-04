# Skill: Analyze Wine Data

## Purpose
Provide a repeatable workflow for exploratory data analysis on the UCI Wine dataset.

## Steps
1. Load the dataset using `sklearn.datasets.load_wine()`.
2. Convert to a polars DataFrame and add the target column.
3. Compute summary statistics, missing values, and class balance.
4. Plot distributions for each feature and a correlation heatmap.
5. Identify outliers using the IQR method.
6. Save plots and summary JSON files to the designated output directory.

## Output Artifacts
- `summary_stats.json`
- `missing_values.json`
- `class_balance.json`
- `outlier_counts.json`
- `distributions.png`
- `correlation_matrix.png`
- `class_balance.png`
