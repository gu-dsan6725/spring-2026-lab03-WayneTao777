# Exercise 2: Configure Advanced Hooks

## Objective

Configure three types of Claude Code hooks that automate quality checks during development:
a command hook, a prompt-based hook, and an agent-based hook.

## Background

Claude Code hooks are shell commands or LLM prompts that execute automatically at specific
points in the development lifecycle. They are configured in `.claude/settings.json` and
guarantee execution (unlike instructions in CLAUDE.md which are suggestions).

Hook types:
- **Command hooks** (`type: "command"`): Run a shell script
- **Prompt hooks** (`type: "prompt"`): Send a prompt to an LLM for evaluation
- **Agent hooks** (`type: "agent"`): Spawn a subagent with tool access to verify conditions

Reference: [Claude Code Hooks Documentation](https://code.claude.com/docs/en/hooks)

## Requirements

### Hook 1: Auto-run pytest (Command Hook)

Create a `PostToolUse` hook that automatically runs pytest after any Python file is written
or edited.

**Configuration** (add to `.claude/settings.json`):
- Event: `PostToolUse`
- Matcher: `Write|Edit`
- Type: `command`
- Script: `scripts/run_tests.sh`

**Script requirements** (`scripts/run_tests.sh`):
1. Read the JSON input from stdin
2. Extract `file_path` from `tool_input`
3. Only run if the file is a `.py` file
4. Execute `uv run pytest` with the `--tb=short` flag
5. If tests fail (exit code non-zero), write the failure output to stderr and exit with code 2
6. If tests pass, exit with code 0

### Hook 2: Check for Tests Before Stopping (Prompt Hook)

Create a `Stop` hook that uses an LLM to verify Claude has written tests before finishing.

**Configuration**:
- Event: `Stop`
- Type: `prompt`
- Prompt: Write a prompt that instructs the LLM to examine the conversation transcript
  and determine if test files were created for any new Python code. The LLM should return
  `{"ok": false, "reason": "..."}` if tests are missing.

### Hook 3: Verify CLAUDE.md Compliance (Agent Hook)

Create a `PostToolUse` hook that spawns an agent to verify newly written Python files
follow the coding standards defined in CLAUDE.md.

**Configuration**:
- Event: `PostToolUse`
- Matcher: `Write|Edit`
- Type: `agent`
- Prompt: Write a prompt that instructs the agent to read the modified Python file and
  check for: type annotations on parameters, proper logging configuration, private
  functions starting with underscore, and functions under 50 lines. The agent can use
  Read and Grep tools to inspect the file.

## Deliverables

1. Updated `.claude/settings.json` with all three hooks configured
2. `scripts/run_tests.sh` (executable)
3. A brief writeup (in a markdown file) explaining:
   - When each hook fires and why you chose that event
   - The difference between command, prompt, and agent hooks
   - One limitation of each hook type

## Grading Criteria

- Correct JSON syntax in settings.json
- Script properly reads stdin JSON and handles exit codes
- Prompt hook includes clear evaluation criteria
- Agent hook prompt references specific CLAUDE.md standards
- Writeup demonstrates understanding of hook types and trade-offs
