# Exercise 1: Rules and Workflows in Google Antigravity

## Objective

Create project-level rules and a custom workflow in Google Antigravity to guide the agent
when building an EDA pipeline.

## Background

Google Antigravity uses two key customization mechanisms:

- **Rules**: Always-on instructions that guide agent behavior. Stored in `.agent/rules/`
  for project scope or `~/.gemini/GEMINI.md` for global scope. Rules are the equivalent
  of Claude Code's CLAUDE.md -- they act as system instructions the agent always follows.

- **Workflows**: Saved prompts triggered on demand with `/workflow-name`. Stored in
  `.agent/workflows/`. Workflows are the equivalent of Claude Code's slash commands --
  explicit actions you invoke when needed.

Reference:
- [Customize Antigravity with rules and workflows](https://atamel.dev/posts/2025/11-25_customize_antigravity_rules_workflows/)
- [Getting Started with Google Antigravity](https://codelabs.developers.google.com/getting-started-google-antigravity)

## Requirements

### 1. Create a Data Quality Rule

Create `.agent/rules/data-quality.md` with rules that instruct the agent to:
- Always check for missing values before any analysis
- Always check for duplicate rows and report them
- Always log the shape of any DataFrame after loading or transforming it
- Never drop rows without logging how many were removed and why
- Always use polars instead of pandas

### 2. Create an EDA Workflow

Create `.agent/workflows/run-full-eda.md` that defines a step-by-step workflow for
comprehensive EDA:

1. Load the California Housing dataset
2. Apply data quality checks (as defined in the rules)
3. Compute summary statistics per feature
4. Generate distribution plots
5. Create a correlation heatmap
6. Identify the top 3 features most correlated with the target variable
7. Generate a scatter plot of each top feature vs target
8. Save all artifacts to `output/`

### 3. Run the Workflow

1. Open Google Antigravity in the project directory
2. The agent should automatically follow your data quality rules
3. Invoke the workflow with `/run-full-eda`
4. Verify the output includes all required plots and statistics

## Deliverables

1. `.agent/rules/data-quality.md`
2. `.agent/workflows/run-full-eda.md`
3. The generated `01_eda.py` file
4. Output artifacts in `output/`
5. A brief comparison (1 paragraph) of how writing rules in Antigravity compares
   to writing CLAUDE.md in Claude Code

## Grading Criteria

- Rules are clear, specific, and actionable
- Workflow steps are ordered logically with clear expected outputs
- Generated code follows the rules (polars, logging, quality checks)
- Comparison demonstrates understanding of both tools
