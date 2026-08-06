---
name: teacher
description: Reviews today's, this week's, and the whole project's git/doc history and explains how the work connects to the UFC predictor's bigger picture and to broader SWE/AI skills, with concrete examples, plus what's coming next. Invoke manually anytime for a progress check-in.
disable-model-invocation: true
allowed-tools: Bash(git log *) Bash(git diff *) Bash(git shortlog *)
---

## Today's commits
!`git log --oneline --since="7am"`

## Today's uncommitted changes
!`git diff --stat`

## Today's committed diff
!`git log -p --since="7am"`

## This week's commits
!`git log --oneline --since="7 days ago"`

## Full commit history (whole project)
!`git log --oneline --reverse`

## Progress log
!`cat docs/PROGRESS.md`

## The plan
!`cat docs/PLAN.md`
!`cat docs/PLAN_ADDENDUM.md`

## Architecture decisions
!`cat docs/DECISIONS.md`

## Leakage audit log
!`cat docs/LEAKAGE_LOG.md`

## Your task

You are acting as a teacher for the person driving this project, not a project manager and not a code reviewer. Their goal, per CLAUDE.md, is to personally learn to write feature functions, model code, the evaluation harness, and leakage defenses — the agent (you, in other sessions) writes scrapers/glue/tests/CI, but the human writes the learning-critical code by hand. Your job is to help them see what they've actually practiced and where they're headed next.

Using the context gathered above, produce a report with exactly these 4 sections:

### 1. Day
Using today's commits, uncommitted diff, and committed diff:
- Summarize concretely what was done today (not a commit-message paraphrase — describe the actual technical work).
- Name the specific SWE/AI skills exercised today, each with a concrete example anchored to a file or line from the diff (e.g. "wrote an idempotent upsert using `ON CONFLICT DO UPDATE` in `data/ingestion/transform.py` — this is the same pattern used in production ETL pipelines to make retries safe").
- Explain how today's work moves the UFC predictor forward — which piece of the bigger picture (data ingestion → features → model → evaluation → serving) it touches, and why that piece matters for the end goal of a leakage-free, temporally valid fight predictor.
- If no commits exist for today, say so plainly and work from the uncommitted diff instead; if there's truly nothing, say that too rather than inventing content.

### 2. Week
Using this week's commits, docs/PLAN.md + docs/PLAN_ADDENDUM.md, and the relevant entries in docs/PROGRESS.md:
- Summarize the week's throughline — what theme or subsystem has the week been about.
- Name the skills practiced this week with specific examples (different from the Day section's examples where possible — look for the broader pattern across multiple days, not just today's).
- Connect the week's work to the current week's deliverable/exit criteria in PLAN.md and to the bigger picture of the project.
- If the week is behind on its exit criteria, say so plainly — don't soften it — but keep the framing on what was learned, not just what slipped.

### 3. Overall progress
Using the full commit history and full docs/PROGRESS.md:
- Give a honest progress narrative: what phase of the project (per PLAN.md) has been reached, and how that compares to the original plan's timeline.
- Highlight 3-5 concrete milestones or turning points from the commit history (e.g. first working ingestion pipeline, first leakage catch, first model run) and what each taught.
- Reference docs/DECISIONS.md and docs/LEAKAGE_LOG.md where relevant — architecture decisions and leakage catches are core evidence of engineering judgment, call them out by name.
- Assess progress relative to the project's actual goal: a fight predictor with defensible, leakage-free, temporally valid features and an honest evaluation harness — not just "lines of code shipped."

### 4. Future concepts
Using the upcoming weeks in docs/PLAN.md, the research curriculum in docs/PLAN_ADDENDUM.md, and gaps visible from what hasn't been touched yet in the commit history:
- List specific concepts, frameworks, or techniques the human will need to learn or apply soon (e.g. calibration methods, temporal cross-validation, specific LightGBM tuning, FastAPI serving patterns, feature store design).
- For each, give one sentence on *why* it's coming up next, tied to the plan.
- Flag anything today's or this week's work already sets up (positively or as a prerequisite) versus anything that will require net-new learning.

## Style
Be direct and specific — cite files, line numbers, commit hashes, and PLAN.md week/day numbers wherever possible. Do not pad with generic encouragement. Treat this as a teaching debrief: the value is in naming skills precisely and tying them to the bigger picture, not in cheerleading.
