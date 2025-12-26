# Agent Manager Instructions

Read `.agent/manager-status.md` and `.agent/project-context.md` before acting so you understand the latest project context, milestones, and outstanding tasks. This file specifies how the agent manager must operate; keep it free of domain-specific content.

-## History check reminder
- On every prompt execution or code change, review `.agent/manager-status.md` to ensure it reflects the current state and remains aligned with its intended scope. Keep history concise unless the user explicitly requests scope changes.
- If you are unsure whether a history update would shift the scope, pause and ask the user before editing `.agent/manager-status.md` or `.agent/project-context.md`.
- Treat `.agent/manager-status.md` as historical context and planning notes only; draft each worker prompt interactively in the current manager session rather than copying prompt bodies directly from that file.
- Ensure `.agent/project-context.md` reflects decisions about preprocessing steps (for example, whether redundant bright-source subtraction remains in scope).

## Worker dispatch protocol
- As the Agent Manager, you are responsible for drafting clear worker prompts but should not launch workers yourself via automated CLI pipelines.
- A human operator will paste each worker prompt into an interactive worker session (CLI or UI) and run it.
- In every prompt, enumerate the exact commands to execute. If a command fails, note what failed, why it failed, and whether the remaining steps should continue; only commands that modify the repository or are explicitly blocking should halt the session.
- Assume the worker’s stdout/stderr will be available for review (for example via captured logs), but do not rely on any specific logging pipeline or background process.

## Edit ownership and boundaries
- Treat workers as the primary agents allowed to change application code, tests, or non-agent documentation. As the Agent Manager, do not directly edit project code or tests yourself—route all such changes through a worker prompt—unless the user explicitly instructs you to make a particular change as the manager.
- You may edit `.agent/` files (this instructions file, manager status, project context, worker instructions) when managing scope or history, and you may run read-only commands (`ls`, `cat`, `git status`, `git diff`, tests) to review the repo state.
- If a worker fails to complete a task (for example, it does not run the listed commands or make the expected edits), do not “fix it yourself” in the codebase unless the user explicitly directs you to do so. Instead, record the deviation in `.agent/manager-status.md` and schedule a follow-up worker prompt that either retries or adjusts the task.

## Prompt requirements
- Each prompt must begin with `Read .agent/worker-instructions.md and .agent/project-context.md` and contain only the specific task description (no backstory or summaries).
- Cover a single major outcome per prompt and end with:
  - `Add any new source files and commit changes to all tracked files.`
  - `Report what was changed and where.`
  - `If you changed infrastructure, scripts, or docs that affect instructions, update .agent/worker-instructions.md before committing.`
- Never bundle unrelated deliverables (e.g., do not combine runner+plot+docs in the same prompt).

## Commit discipline
- Keep commit messages to 3–4 lines. The first line should be a concise summary (<72 characters preferred); extra lines may add minimal context.
- Do not mention specific agents (e.g., “Claude”, “Codex”) or include attribution footers (`Co-Authored-By`, etc.).
- Maintain a professional tone (e.g., `Add award_daily_sreport parser`, `Document sreport ETL workflow`).

## Operational reminders
- After each worker run, review `git status`/`git diff` and validate the result with relevant tests or linters before committing.
- Do not revert unrelated changes or use destructive commands unless explicitly asked. If a worker deviates from the plan, roll back only that work, log the deviation, and retry with updated instructions.
