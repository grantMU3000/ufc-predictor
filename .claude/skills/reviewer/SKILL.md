---
name: reviewer
description: Reviews the current diff specifically for data leakage patterns, then does a normal code review pass. Invoke manually before opening a PR.
disable-model-invocation: true
allowed-tools: Bash(git diff *)
---

## Current diff
!`git diff main`

## Your task
Review the diff above for data leakage patterns first, then general code quality:

1. Any feature function using data from after `as_of_date`
2. Any train/test split that isn't strictly temporal
3. Betting odds used as a model feature (should only appear in baseline/backtest code)
4. Any feature whose distribution suspiciously differs between train and validation

Flag each finding with file and line. Then do a normal review pass: readability, test coverage, error handling.