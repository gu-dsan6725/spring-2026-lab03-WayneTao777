# Skill: Wine Profiler

## Purpose
Profile the UCI Wine dataset with class-specific statistics and discriminative feature analysis.

## Steps
1. Load the Wine dataset and convert to polars DataFrame.
2. Compute overall summary statistics and per-class summary statistics.
3. Plot distributions per class for key features.
4. Compute feature means per class and identify top differentiating features.
5. Save a short markdown summary and any plots to `output/antigravity/`.

## Output Artifacts
- `wine_profile.md`
- `class_feature_means.json`
- Optional plots per class
