---
name: planner
description: Reads PLAN.md, PROGRESS.md, and open GitHub issues, then proposes today's 3 tasks. Invoke manually each morning.
disable-model-invocation: true
allowed-tools: Bash(gh issue list *)
---

## Yesterday's progress
!`tail -30 docs/PROGRESS.md`

## Open issues
!`gh issue list --limit 10`

## The plan
See docs/PLAN.md for the current week's deliverable, exit criteria, and research curriculum.

## Your task
Propose exactly 3 tasks for today, each with a rough time estimate, that move toward this week's exit criteria in docs/PLAN.md. Base them on what's still open from yesterday's progress entry and any relevant GitHub issues. Flag anything that looks blocked. Also, give me the research topic for today. Keep the list short — I'll approve or edit it before starting. 