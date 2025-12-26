Agent Worker Instructions
=====================================================

Read this file and `.agent/project-context.md` at the start of every Agent worker session. It defines how a worker should behave when executing a single prompt.

1. Scope & allowed files
------------------------
- Treat each prompt as your entire scope. Do not infer extra tasks from other
  files or background knowledge.
- Do not read `.agent/manager-status.md` unless the prompt explicitly instructs
  you to. That file is for the Agent Manager.
- Only read other `.agent/*` files when the
  prompt tells you to; rely on the prompt for task context.
- Unless explicitly told otherwise, do not edit `.agent/` outside the single
  document you were asked to update.

2. Safety & workflow
--------------------
- Always operate from the repo root unless the prompt specifies another working
  directory.
- Prefer minimal, targeted changes that address the requested task; avoid
  opportunistic refactors or unrelated cleanups.
- Avoid destructive cleanup (no `git reset --hard`, no mass deletions) unless
  explicitly requested. If unexpected files or errors appear, stop and report.
- Run only the commands listed in your prompt. If a command that modifies the
  repository or is marked as blocking fails, stop and report rather than
  guessing follow-up steps. If a system utility (e.g., `rg`) is missing or
  fails, log the failure and continue with an alternative command specified in
  the prompt.
- When a prompt instructs you to “implement”, “extend”, or “modify” code or
  tests, you are expected to actually edit the referenced files (using the
  editing tools available in your environment) rather than only describing
  changes in prose.

3. Failure reporting
--------------------
- If you cannot complete a requested step (for example due to missing tools,
  permission issues, or internal limits), stop and explicitly state:
  * Which step failed.
  * Which of the requested commands were run successfully.
  * Which edits, if any, were applied before stopping.
- Do not silently skip required edits or commands. The Agent Manager relies on
  your final summary to decide whether to retry, adjust the prompt, or roll
  back changes.

4. Validation
-------------
- When tests or scripts are requested, run exactly what the prompt lists (for
  example a specific test command, linter, or CLI tool named in the prompt).
- Treat failing tests as a blocker: do not ignore failures unless the prompt
  explicitly accepts them or provides alternate success criteria.

5. Commit & reporting expectations
----------------------------------
- If the prompt instructs you to commit, use commit messages that:
  * Are 3–4 lines max.
  * Have a concise first line (<72 characters preferred).
  * Include no agent names (“Claude”, “Codex”, etc.) and no attribution footers
    such as `Co-Authored-By`.
- When summarizing work, call out:
  * Which files changed and why.
  * Any tests or scripts run and their outcomes.
  * Any follow-up steps or uncertainties that the Agent Manager should review.

Follow these instructions on every worker session. If you need to revise this
guidance, do so carefully and keep a log of why the change was made.

If you introduced infrastructure, scripts, or docs that affect the project context
or operational guidance, update `.agent/project-context.md` (not this file) before committing.
