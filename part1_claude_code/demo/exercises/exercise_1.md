# Exercise 1: Create a Custom Skill

## Objective

Create a custom Claude Code skill called `/generate-report` that generates a comprehensive
markdown report comparing model performance across different hyperparameter configurations.

## Background

Claude Code skills are markdown files stored in `.claude/skills/<skill-name>/SKILL.md` that
teach Claude reusable workflows. Skills have YAML frontmatter that controls when and how they
are invoked. When you type `/generate-report` in Claude Code, it loads the skill content and
follows its instructions.

Reference: [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)

## Requirements

### 1. Create the Skill Directory

Create the following structure:

```
.claude/skills/generate-report/
    SKILL.md
    templates/
        report_template.md
```

### 2. Write SKILL.md

Your `SKILL.md` must include:

**Frontmatter** (between `---` markers):
- `name`: `generate-report`
- `description`: A clear description of what the skill does and when to use it
- `argument-hint`: `[model-output-dir]` to hint that it expects a directory path

**Instructions** (markdown body):
The skill should tell Claude to:
1. Read the evaluation metrics from `output/evaluation_report.md`
2. Load the trained model and compute additional statistics
3. Generate a report using the template in `templates/report_template.md`
4. Include a metrics comparison table, feature importance ranking, and recommendations
5. Save the report to `output/full_report.md`

### 3. Create the Report Template

Create `templates/report_template.md` with sections for:
- Executive Summary
- Dataset Overview
- Model Configuration
- Performance Metrics (table format)
- Feature Importance (top 5)
- Recommendations for Improvement

### 4. Test the Skill

1. Run the solved examples first to generate model artifacts:
   ```bash
   uv run python part1_claude_code/solved/01_eda.py
   uv run python part1_claude_code/solved/02_feature_engineering.py
   uv run python part1_claude_code/solved/03_xgboost_model.py
   ```
2. Open Claude Code and type `/generate-report output/`
3. Verify the generated report contains all required sections

## Deliverables

1. `.claude/skills/generate-report/SKILL.md` with proper frontmatter and instructions
2. `.claude/skills/generate-report/templates/report_template.md`
3. The generated `output/full_report.md` from running the skill

## Grading Criteria

- Correct YAML frontmatter syntax (name, description, argument-hint)
- Clear, step-by-step instructions in the skill body
- Template includes all required sections
- Skill can be invoked with `/generate-report` and produces the expected output
