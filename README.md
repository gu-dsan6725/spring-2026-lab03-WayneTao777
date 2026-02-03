# AI-Assisted Coding Lab

In this lab you will use two AI-powered coding assistants -- **Claude Code** and **Google Antigravity** -- to build a complete data analytics and machine learning pipeline. The goal is to learn how these tools extend beyond simple code generation through features like hooks, skills, subagents, rules, and workflows.

## Learning Objectives

- Use an AI coding assistant to plan, build, and test a full ML project
- Configure **CLAUDE.md** and **GEMINI.md** to guide AI behavior with project-specific rules
- Create **hooks** that automate quality checks (linting, syntax validation) on every file change
- Build **custom skills** that teach the AI reusable workflows (data analysis, model evaluation)
- Use **subagents** to decompose complex tasks into isolated, parallel work streams
- Compare how Claude Code and Google Antigravity approach the same concepts differently

## Problem Statement

Your task is to build a complete ML pipeline for classifying wines into 3 classes using the **UCI Wine** dataset (`sklearn.datasets.load_wine()`). The pipeline must include:

1. Exploratory data analysis (EDA) using polars and matplotlib
2. Feature engineering and data preparation for modeling
3. **XGBoost** classification model training with cross-validation
4. Model evaluation and a comprehensive performance report

You will complete this task twice -- once using Claude Code and once using Google Antigravity -- to understand how each tool's customization features support the development workflow. Each part includes a `demo/` folder with a reference implementation on the California Housing dataset (regression), so you can see a working example without being able to copy it directly for your classification task.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager installed
- Claude Code CLI installed (for Part 1)
- Google Antigravity IDE installed (for Part 2)

## Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd ai-assisted-coding

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies with uv
uv sync

# Verify the setup
uv run python -c "from sklearn.datasets import load_wine; print('Dataset OK')"
```

## Lab Structure

### Part 1: Claude Code

Located in `part1_claude_code/`. A single guided lab where you use Claude Code to build the full ML pipeline from scratch. The task naturally exercises every major feature:
- **CLAUDE.md**: Project-level instructions enforced on every file write
- **Hooks**: Automated quality checks (ruff, py_compile, test scaffolding) that fire automatically
- **Skills**: `/analyze-data`, `/evaluate-model`, `/generate-report` for reusable workflows
- **Subagents**: Explore, Plan, and Bash agents for task decomposition
- **Slash commands**: `/plan` for structured planning before building

See [part1_claude_code/README.md](part1_claude_code/README.md) for the full walkthrough.

### Part 2: Google Antigravity

Located in `part2_antigravity/`. A single guided lab where you use Google Antigravity to build the full ML pipeline from scratch. The task naturally exercises every major feature:
- **GEMINI.md + Rules**: Project rules that guide agent behavior
- **Workflows**: `/run-eda` and `/train-model` for on-demand step-by-step instructions
- **Skills**: Custom data profiling capability
- **Manager View**: Multi-agent orchestration for parallel task execution

See [part2_antigravity/README.md](part2_antigravity/README.md) for the full walkthrough.

## The AI-Assisted Development Workflow

Both parts follow the same development process:

1. **Plan**: Ask the AI assistant to create an implementation plan in a markdown file
2. **Review**: Read the plan, provide feedback, and request changes
3. **Build**: Have the AI execute the plan, writing code according to your project rules
4. **Test**: Hooks and automation catch issues automatically during development
5. **Iterate**: Review outputs, request improvements, and refine

This mirrors how professional developers work with AI coding assistants in practice.

## Concept Mapping

| Concept | Claude Code | Google Antigravity |
|---|---|---|
| Project instructions | `CLAUDE.md` | `GEMINI.md` + `.agent/rules/` |
| Reusable AI capabilities | Skills (`.claude/skills/`) | Skills (`.agent/skills/`) |
| On-demand commands | Slash commands (`.claude/commands/`) | Workflows (`.agent/workflows/`) |
| Automated checks | Hooks (PreToolUse, PostToolUse) | No native equivalent (use pre-commit) |
| Task decomposition | Subagents (Explore, Plan, Bash) | Manager View (multi-agent orchestration) |
| Execution control | Plan mode | Terminal execution policies (Off, Auto, Turbo) |

## Grading

Each part is worth **50 points** (100 total).

### Part 1: Claude Code (50 points)

If you have access to Claude Code, follow the full walkthrough in [part1_claude_code/README.md](part1_claude_code/README.md) and submit:
- The generated pipeline scripts in `part1_claude_code/src/`
- Output artifacts in `output/`
- The completed Feature Summary Checklist from the README

**If you do not have access to Claude Code**, complete the config-only exercise instead:

1. **Read all existing configuration files** (10 points): Read and understand the existing `CLAUDE.md` files, `.claude/settings.json` (hooks), `.claude/skills/` (skills), and `.claude/commands/plan.md` (slash command). Study the demo solved code in `part1_claude_code/demo/solved/` to see how the configuration influences generated code.

2. **Create a CLAUDE.md for Wine classification** (10 points): Write `part1_claude_code/CLAUDE.md` tailored to the Wine classification task. It should include dataset-specific instructions (3 wine classes, classification metrics, stratified splits) while keeping the existing coding standards.

3. **Create a hook configuration** (10 points): Write a `.claude/settings.json` that defines hooks for the Wine pipeline. Include a PostToolUse command hook for ruff and py_compile, a PostToolUse prompt hook for test file creation, and a PreToolUse command hook to block force pushes. Explain in comments or a companion markdown file what each hook does and when it fires.

4. **Create skills for Wine classification** (10 points): Write skill files in `.claude/skills/` for at least two skills relevant to Wine classification (e.g., `analyze-wine-data`, `evaluate-classifier`). Each skill should have a `SKILL.md` with clear step-by-step instructions the agent would follow.

5. **Create a slash command** (10 points): Write a `.claude/commands/plan-wine.md` slash command that would generate an implementation plan for the Wine classification pipeline. The command should include a template with sections for objectives, architecture, file structure, and implementation steps.

### Part 2: Google Antigravity (50 points)

If you have access to Google Antigravity, follow the full walkthrough in [part2_antigravity/README.md](part2_antigravity/README.md) and submit:
- The generated pipeline scripts in `part2_antigravity/src/`
- Output artifacts in `output/`
- The completed Feature Summary Checklist from the README

**If you do not have access to Google Antigravity**, complete the config-only exercise instead:

1. **Read all existing configuration files** (10 points): Read and understand `.gemini/GEMINI.md`, `.agent/rules/code-style-guide.md`, `.agent/workflows/`, and `.agent/skills/data-profiler/`. Study the demo solved code in `part2_antigravity/demo/solved/`.

2. **Create rules for Wine classification** (10 points): Write `.gemini/GEMINI.md` and `.agent/rules/data-quality.md` tailored to Wine classification. Rules should cover dataset-specific instructions, classification-appropriate metrics, and data quality checks.

3. **Create workflows for Wine classification** (10 points): Write `.agent/workflows/run-wine-eda.md` and `.agent/workflows/train-wine-classifier.md` with step-by-step instructions for EDA and model training on the Wine dataset. Each workflow should reference the rules and include quality check steps.

4. **Create a skill for Wine classification** (10 points): Write `.agent/skills/wine-profiler/SKILL.md` that instructs the agent to profile the Wine dataset, including class-specific statistics, feature distributions per class, and discriminative feature analysis.

5. **Set up pre-commit hooks and write a comparison** (10 points): Create a `.pre-commit-config.yaml` with ruff hooks. Write a brief comparison (300-500 words) answering: How do Claude Code hooks compare to using pre-commit + Antigravity rules? What are the advantages and limitations of each approach?

## Quick Reference

```bash
# Install dependencies
uv sync

# Run a solved example (reference implementations in demo/)
uv run python part1_claude_code/demo/solved/01_eda.py
uv run python part2_antigravity/demo/solved/01_eda.py

# Lint all Python files
uv run ruff check .

# Format all Python files
uv run ruff format .

# Compile-check a file
uv run python -m py_compile part1_claude_code/demo/solved/01_eda.py
```
