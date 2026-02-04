# Claude Code Hooks vs Antigravity Rules and Pre-Commit

Claude Code and Google Antigravity both support “always-on” guidance, but they enforce quality in
very different ways. Claude Code uses hooks that run automatically on file write or command
execution. That means a Python file saved by the agent immediately triggers ruff, formatting, and
syntax checks. The enforcement is procedural and automatic. In practice, this reduces drift: even if
a model “forgets,” the hook still runs. The biggest advantage is reliability: quality checks happen
at the exact moment code is created, not later. The limitation is portability. Hooks are tied to
Claude Code’s lifecycle, so the same enforcement does not automatically carry to other tools or
editors.

Antigravity’s rules (GEMINI.md and rule files) are different. They act like persistent guidance for
the agent, but they are not enforced by the tool. In other words, they can influence behavior, but
they do not guarantee execution. This gives flexibility and makes it easy to express richer
instructions, but it relies on the agent to comply. That is why Antigravity often pairs rules with
workflows and external enforcement, such as pre-commit hooks.

Pre-commit sits at the git boundary. It ensures code quality at commit time, not at write time. The
advantage is that it is tool-agnostic and portable across editors and IDEs. It also integrates into
standard team workflows. The main limitation is timing: issues are caught later, after code is
written, and only if the user commits. This can allow low-quality intermediate states to persist
until the commit.

In the Wine classification lab, Claude Code’s hooks provide immediate guardrails. Every file write
runs ruff and py_compile, which enforces style and prevents syntax errors from landing in the repo.
Antigravity’s rules, by contrast, ensure the agent knows it should use polars, log correctly, and
keep functions short, but they do not force compliance. Adding pre-commit to Antigravity closes part
of the enforcement gap, but the checks happen later and only on commit. In summary, Claude Code’s
hooks are stronger at immediate correctness, while Antigravity’s rules are stronger at flexible
policy definition. Pre-commit helps Antigravity bridge the enforcement gap, but it cannot fully
match the “on every write” reliability of Claude Code.
