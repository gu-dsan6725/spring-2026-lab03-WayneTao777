#!/bin/bash
# PostToolUse hook: runs ruff and py_compile on Python files after Write/Edit
# This script is called by Claude Code after any file is written or edited.
# It receives JSON on stdin with tool_input.file_path.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only process Python files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# Check if file exists
if [[ ! -f "$FILE_PATH" ]]; then
    exit 0
fi

ERRORS=""

# Run ruff check with auto-fix
RUFF_OUTPUT=$(uv run ruff check --fix "$FILE_PATH" 2>&1)
RUFF_EXIT=$?
if [[ $RUFF_EXIT -ne 0 ]]; then
    ERRORS="${ERRORS}Ruff errors:\n${RUFF_OUTPUT}\n"
fi

# Run ruff format
uv run ruff format "$FILE_PATH" 2>&1

# Run py_compile to check syntax
COMPILE_OUTPUT=$(uv run python -m py_compile "$FILE_PATH" 2>&1)
COMPILE_EXIT=$?
if [[ $COMPILE_EXIT -ne 0 ]]; then
    ERRORS="${ERRORS}Syntax error:\n${COMPILE_OUTPUT}\n"
fi

# Report results
if [[ -n "$ERRORS" ]]; then
    echo -e "$ERRORS" >&2
    exit 2
fi

exit 0
