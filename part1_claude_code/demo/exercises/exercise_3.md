# Exercise 3: End-to-End with Subagents and Plan Mode

## Objective

Use the full AI-assisted development workflow -- plan, review, build, test -- to add
cross-validation and hyperparameter tuning to the existing XGBoost pipeline. You will
use the `/plan` slash command, review and modify the plan, and then have Claude build
the implementation using subagents.

## Background

Claude Code supports multiple types of subagents that run as isolated processes with their
own context windows:

- **Explore** agent: Read-only tools (Read, Grep, Glob) for codebase exploration
- **Plan** agent: Designs implementation approaches based on exploration results
- **Bash** agent: Executes shell commands
- **general-purpose** agent: Full tool access for complex tasks

When Claude uses the Task tool, it spawns a subagent. You can encourage Claude to use
subagents by asking it to "explore the codebase first" or "use subagents to investigate."

Reference: [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)

## The Workflow

### Step 1: Create the Plan

Use the `/plan` slash command to create an implementation plan:

```
/plan Add cross-validation and hyperparameter tuning to the XGBoost pipeline.
Use GridSearchCV or RandomizedSearchCV from scikit-learn. The tuned model
should be compared against the baseline model from 03_xgboost_model.py.
Save comparison results to output/tuning_report.md.
```

Claude will write a plan to `.scratchpad/plan.md`.

### Step 2: Review and Modify the Plan

1. Open `.scratchpad/plan.md` and review the proposed steps
2. Make at least TWO modifications to the plan. For example:
   - Change the cross-validation strategy (e.g., use 5-fold instead of 3-fold)
   - Add a specific hyperparameter range to search
   - Request a specific visualization
3. Tell Claude about your changes and ask it to update the plan

### Step 3: Build with Subagents

Ask Claude to implement the updated plan. Encourage it to:
- First use an **Explore** subagent to understand the existing code structure
- Then implement the changes based on what it learned
- Use subagents for parallel tasks (e.g., one for the tuning code, one for tests)

Suggested prompt:
```
Implement the plan in .scratchpad/plan.md. Start by using subagents to explore
the existing code in solved/ to understand the current pipeline structure.
Then build the implementation following the plan.
```

### Step 4: Verify with Hooks

After Claude builds the code:
1. The PostToolUse hook should automatically run ruff and py_compile on every file
2. Check that the hook output shows no errors
3. Run the new code manually to verify it works:
   ```bash
   uv run python part1_claude_code/solved/04_hyperparameter_tuning.py
   ```

## Deliverables

1. `.scratchpad/plan.md` -- the original plan created by Claude
2. `.scratchpad/plan_v2.md` -- the modified plan after your review
3. `solved/04_hyperparameter_tuning.py` -- the implementation
4. `output/tuning_report.md` -- comparison of baseline vs tuned model
5. A brief writeup answering:
   - How did subagents help decompose the task?
   - What did the Explore subagent discover about the existing code?
   - Did the hooks catch any issues during development?
   - How would you improve the plan-review-build workflow?

## Grading Criteria

- Plan is detailed and follows the template structure
- At least two meaningful modifications to the plan
- Implementation follows CLAUDE.md coding standards
- Tuning report includes a clear comparison table
- Writeup shows understanding of the subagent and plan-review workflow
