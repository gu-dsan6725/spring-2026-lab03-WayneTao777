#!/bin/bash
# PreToolUse hook: blocks git push --force commands
# This script is called by Claude Code before any Bash command executes.
# It receives JSON on stdin with tool_input.command.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Check for force push patterns
if echo "$COMMAND" | grep -qE 'git\s+push\s+.*(-f|--force)'; then
    echo '{"decision":"block","reason":"Force push is blocked by project hooks. Use regular git push instead."}'
    exit 0
fi

exit 0
