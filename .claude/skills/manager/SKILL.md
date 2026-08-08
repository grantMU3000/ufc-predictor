---
name: manager
description: Compares actual progress against PLAN.md exit criteria and names what to cut if behind. Invoke manually on Fridays.
disable-model-invocation: true
---

## This week's plan
!`cat docs/PLAN.md`
!`cat docs/PLAN_ADDENDUM.md`

## This week's progress entries
!`tail -100 docs/PROGRESS.md`

## Your task
Compare actual progress this week against the current week's deliverable and exit criteria in docs/PLAN.md and docs/PLAN_ADDENDUM.md. Be direct: is the exit criterion realistically going to be met by the deadline? If not, name specifically what should be cut from *next* week, per the plan's own rule — cut scope from the next week, don't extend the current one. Do not soften or downplay slippage.