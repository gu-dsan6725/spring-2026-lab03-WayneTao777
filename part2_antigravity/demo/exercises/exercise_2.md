# Exercise 2: Custom Skills in Google Antigravity

## Objective

Create two custom skills for Google Antigravity: a data profiling skill and a model
evaluation skill. Use them to analyze the California Housing dataset and evaluate the
trained XGBoost model.

## Background

In Google Antigravity, a skill is a directory-based package containing a `SKILL.md`
definition file and optional supporting assets (scripts, templates, references). Skills
are stored in:

- `~/.gemini/antigravity/skills/` for global skills (available across all projects)
- `<workspace>/.agent/skills/` for project-specific skills

When a user asks the agent to perform a task, the agent scans its available skills. If it
finds a matching skill, it loads those instructions into its context window. Unlike rules
(which are always active), skills are loaded on demand when relevant.

This is the same concept as Claude Code skills (`.claude/skills/<name>/SKILL.md`), with a
different directory structure.

Reference:
- [How to Build Custom Skills in Google Antigravity](https://medium.com/google-cloud/tutorial-getting-started-with-antigravity-skills-864041811e0d)

## Requirements

### 1. Create a Feature Analysis Skill

Create `.agent/skills/feature-analysis/SKILL.md` that instructs the agent to:

1. Load a dataset (passed as an argument)
2. For each numeric feature, compute:
   - Distribution shape (skewness, kurtosis)
   - Percentage of zeros
   - Percentage of outliers (IQR method)
3. For the target variable:
   - Distribution plot
   - Summary statistics
4. Rank features by their correlation with the target
5. Generate a feature importance summary table
6. Save results to `output/feature_analysis.md`

Include a `description` in the skill so the agent knows when to use it automatically.

### 2. Create a Model Comparison Skill

Create `.agent/skills/model-comparison/SKILL.md` that instructs the agent to:

1. Train two models on the same data:
   - XGBoost with default parameters
   - XGBoost with tuned parameters (the agent should determine reasonable ranges)
2. Evaluate both on the test set using RMSE, MAE, and R-squared
3. Create a comparison table
4. Generate a side-by-side residual plot
5. Recommend which model to use and why
6. Save the comparison to `output/model_comparison.md`

### 3. Test the Skills

1. Run the solved examples first to generate the data splits
2. Open Antigravity and try asking: "Analyze the features in the California Housing dataset"
   - The feature-analysis skill should activate
3. Try: "Compare different XGBoost configurations"
   - The model-comparison skill should activate

## Deliverables

1. `.agent/skills/feature-analysis/SKILL.md`
2. `.agent/skills/model-comparison/SKILL.md`
3. Generated output files (`feature_analysis.md`, `model_comparison.md`)
4. A brief comparison (1 paragraph) of how skills work in Antigravity vs Claude Code

## Grading Criteria

- Skills have clear descriptions that enable automatic matching
- Instructions are specific enough to produce consistent output
- Both skills follow the project's coding standards
- Comparison identifies at least one similarity and one difference
