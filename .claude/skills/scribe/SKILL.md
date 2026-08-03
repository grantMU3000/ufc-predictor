---
name: scribe
description: Drafts today's PROGRESS.md entry from git log and diff. Invoke manually at end of day.
disable-model-invocation: true
allowed-tools: Bash(git log *) Bash(git diff *)
---

## Today's commits
!`git log --oneline --since="9am"`

## Uncommitted changes
!`git diff --stat`

## Your task
Draft today's docs/PROGRESS.md entry using the template at the top of that file. Fill in "Shipped" from the commits above, leave "Blocked / open questions" and "Energy / notes" for me to fill in myself, and propose a "Tomorrow's first task" based on what's still incomplete. Show me the draft — don't write it into the file yet.