# Plan: Wine Classification Pipeline (Antigravity)

## Objective
Build a Wine classification pipeline with EDA, feature engineering, XGBoost model training with
5-fold stratified cross-validation, and a comprehensive evaluation report.

## Steps
1. Create EDA script in `part2_antigravity/src/01_eda.py`.
   - Load `load_wine()` and convert to polars DataFrame.
   - Compute summary stats, missing values, class balance, distributions, correlation heatmap, outliers.
   - Save artifacts to `output/antigravity/`.
2. Create feature engineering script in `part2_antigravity/src/02_feature_engineering.py`.
   - Add at least 3 derived features.
   - Handle non-finite values, standard scaling, stratified train/test split.
   - Save parquet splits and scaler to `output/antigravity/`.
3. Create model training script in `part2_antigravity/src/03_xgboost_model.py`.
   - Train XGBoost classifier and run 5-fold stratified CV.
   - Compute accuracy, precision, recall, F1-score, and confusion matrix.
   - Save plots and metrics to `output/antigravity/`.
4. Create report generator in `part2_antigravity/src/04_generate_report.py`.
   - Summarize EDA, feature engineering, CV metrics, and test metrics.
   - Save `full_report.md` to `output/antigravity/`.

## Technical Decisions
- Use polars for data manipulation.
- Use XGBoost (xgboost.XGBClassifier).
- Use matplotlib/seaborn for plots.
- Follow rules in `.gemini/GEMINI.md` and `.agent/rules/`.

## Testing Strategy
- Run `ruff check --fix`, `ruff format`, `python -m py_compile` on scripts.
- Execute the pipeline scripts in order and verify artifacts.

## Expected Output
- Plots: distributions, correlation matrix, class balance, confusion matrix, feature importance.
- Artifacts: parquet splits, scaler, model file, JSON metrics, `full_report.md`.
