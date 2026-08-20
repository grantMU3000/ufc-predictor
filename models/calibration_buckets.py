"""
ADR-014's evidence gate — Week 3 Tuesday (docs/PLAN.md Section 3:
"Bucket ECE by total_ufc_fights / days_since_last_fight (ADR-014)").

WHAT THIS DECIDES: ADR-014 deferred Glicko-2 with a written trigger,
not a vague "maybe later." Glicko's rating deviation (RD) tracks how
much is actually KNOWN about a fighter right now — it grows during
inactivity, shrinks with consistent activity. Elo has no equivalent:
a debutant who won one fight and a 20-fight veteran can sit at the
same rating number, and Elo can't tell you it's far less sure about
the first one.

The ADR's claim is that IF that missing uncertainty signal matters,
it should show up as measurably WORSE CALIBRATION for exactly the
populations Elo can't distinguish — low-fight-count fighters, and
fighters returning from a long layoff. This script checks whether
that's true. If it is, RD gets built as a standalone feature. If it
isn't, Glicko goes to IDEAS.md and stays there.

Simple version: the model says "70%" about a lot of fights. Overall,
those fights are won about 70% of the time, so it looks honest. But
maybe it's honest about veterans and wildly overconfident about
debutants, and the two errors cancel out in the average. This script
splits val into experience groups and checks each one separately.

--- PRE-REGISTERED DECISION RULE (written BEFORE seeing results) ---

Baseline: overall tuned-model ECE on FULL val = 0.0235.
(NOT the 0.0318 odds-covered figure — buckets are cut on full val for
sample size, and odds coverage is unrelated to fighter experience.
Using the larger number would lower the bar in our own favor.)

GATE OPENS (build Glicko-2 RD) if BOTH hold:
  1. A bucket with n >= MIN_BUCKET_N shows ECE >= 2x baseline, AND
  2. That bucket is one ADR-014 actually named — a LOW fight-count
     bucket or a LONG layoff bucket. A badly-calibrated 11+ veteran
     bucket is a real finding worth logging, but it is NOT evidence
     for RD, because RD is not the thing that would fix it.

GATE STAYS CLOSED otherwise. A miss is logged to IDEAS.md with these
numbers attached, so a future revisit starts from data.

MIN_BUCKET_N exists because val is 2,034 rows and a 730+ layoff
bucket may hold well under a hundred. ECE on 60 rows is noise wearing
a decimal point. This threshold is fixed here, before running, so the
outcome can't be rationalized after the fact.
"""

from typing import Any

import numpy as np
import pandas as pd

from features.build_lgbm_matrix import build_train_val_with_elo
from models.lightgbm_model import load_tuned_params, train_lightgbm_baseline
from models.metrics import evaluate, expected_calibration_error

# --- Your calls. Change these before running, not after. ---

MIN_BUCKET_N = 150          # below this, report but refuse to interpret
BASELINE_ECE = 0.0235       # tuned LightGBM, full val (docs/RESULTS.md)
GATE_MULTIPLE = 2.0         # bucket ECE >= this x baseline opens the gate

# Fight-count edges: tight at the bottom, wide at the top. That's
# deliberate — the whole question is about the low end, where Elo's
# uncertainty is highest, so that's where resolution is worth
# spending sample size on. Nobody claims a 12-fight and a 30-fight
# veteran differ much in how well-known they are.
FIGHT_COUNT_BINS = [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, np.inf]
FIGHT_COUNT_LABELS = ["0 (debut)", "1-2", "3-5", "6-10", "11-20", "21+"]

# Layoff edges in days. ~120 days is a normal active-fighter turnaround;
# 365+ is where "is this fighter still the same fighter" becomes a real
# question, which is precisely the RD hypothesis.
LAYOFF_BINS = [-0.5, 120, 240, 365, 730, np.inf]
LAYOFF_LABELS = ["<120d", "120-240d", "240-365d", "365-730d", "730d+"]


def _as_dict(result: Any) -> dict:
    """
    Small adapter so this file doesn't care whether models.metrics
    .evaluate() hands back a dict, a Series, or a one-row frame.
    Keeps the gate logic readable instead of littered with accessor
    guesses.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, pd.Series):
        return result.to_dict()
    if isinstance(result, pd.DataFrame):
        return result.iloc[0].to_dict()
    raise TypeError(f"unexpected evaluate() return type: {type(result)}")


def bucket_ece(
    val: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    column: str,
    bins: list,
    labels: list,
) -> pd.DataFrame:
    """
    Splits val into buckets on one column and scores each bucket
    through the SAME evaluate() harness every other number in this
    project came from — no separate ECE implementation, so a bucket's
    number is directly comparable to docs/RESULTS.md.

    Buckets on the self_ column, not the diff_ column, deliberately:
    ADR-014's question is about how much is known about THIS fighter,
    not about the experience gap between the two corners.

    NaN rows (a debutant has no days_since_last_fight) are collected
    into their own "no prior fight" bucket rather than dropped —
    dropping them would hide the single population the gate cares
    most about.

    Parameters
    ----------
    val : pd.DataFrame
        The symmetrized val split, in the same row order the
        predictions were produced from. Row order matters — y_prob is
        positional, so a reindexed val would silently misalign every
        probability with the wrong fight.
    y_true, y_prob : np.ndarray
        Straight from train_lightgbm_baseline.
    column : str
        A self_ column to bucket on.
    bins, labels : list
        pd.cut edges and their names.

    Returns
    -------
    pd.DataFrame — one row per bucket: bucket, n, accuracy, log_loss,
    brier, ece, mean_pred, actual_rate, ece_ratio, interpretable.
    """
    if len(val) != len(y_prob):
        raise ValueError(
            f"val has {len(val)} rows but {len(y_prob)} predictions — "
            f"row alignment is positional here, so this must match."
        )

    raw = val[column]
    assigned = pd.cut(raw, bins=bins, labels=labels)
    assigned = assigned.cat.add_categories(["no prior fight"])
    assigned = assigned.fillna("no prior fight")

    rows = []
    for bucket in list(labels) + ["no prior fight"]:
        mask = (assigned == bucket).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue

        scored = _as_dict(evaluate(y_true[mask], y_prob[mask], name=str(bucket)))
        ece = float(scored["ece"])

        rows.append(
            {
                "bucket": bucket,
                "n": n,
                "accuracy": float(scored["accuracy"]),
                "log_loss": float(scored["log_loss"]),
                "brier": float(scored["brier"]),
                "ece": ece,
                # mean_pred vs actual_rate is the direction check ECE
                # can't give you: ECE is unsigned, so a bucket at
                # 0.05 could be systematically overconfident OR
                # underconfident. RD would only help the first case.
                "mean_pred": float(y_prob[mask].mean()),
                "actual_rate": float(y_true[mask].mean()),
                "ece_ratio": ece / BASELINE_ECE,
                "interpretable": n >= MIN_BUCKET_N,
            }
        )

    return pd.DataFrame(rows)


def gate_verdict(table: pd.DataFrame, gate_buckets: list[str]) -> tuple[bool, list[str]]:
    """
    Applies the pre-registered rule from the module docstring.

    Separated from bucket_ece so the numbers and the judgement stay
    distinct — the table is a measurement, this is the decision, and
    conflating them is how a threshold quietly becomes "whatever the
    data did."

    Parameters
    ----------
    table : pd.DataFrame
        Output of bucket_ece.
    gate_buckets : list[str]
        Only the buckets ADR-014 actually named as RD-relevant (low
        fight count / long layoff). A miscalibrated veteran bucket
        doesn't open this gate — see docstring rule #2.

    Returns
    -------
    (opens, reasons) — whether the gate opens, and the bucket-level
    findings that drove it either way.
    """
    reasons = []
    opens = False

    for row in table.itertuples(index=False):
        if row.bucket not in gate_buckets:
            continue
        if not row.interpretable:
            reasons.append(
                f"{row.bucket}: n={row.n} < {MIN_BUCKET_N} — too small to read "
                f"(ece={row.ece:.4f}, not counted either way)"
            )
            continue
        if row.ece_ratio >= GATE_MULTIPLE:
            opens = True
            reasons.append(
                f"{row.bucket}: ece={row.ece:.4f} = {row.ece_ratio:.2f}x baseline, "
                f"n={row.n} — GATE TRIGGER"
            )
        else:
            reasons.append(
                f"{row.bucket}: ece={row.ece:.4f} = {row.ece_ratio:.2f}x baseline, "
                f"n={row.n} — below trigger"
            )

    return opens, reasons

def check_buckets_against_null(
    table: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    buckets_to_check: list[str],
    n_permutations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Answers one question per bucket: "is this ECE actually bad, or is
    it just what a SMALL group looks like even when nothing's wrong?"

    Simple version: take the whole val set, ignore who's actually a
    debutant or a long-layoff fighter, and just grab a random handful
    of the SAME SIZE as the real bucket. Compute ECE on that random
    handful. Do it 2,000 times. That builds a picture of "how noisy
    does ECE normally look for a group this small, purely from bad
    luck." Then check where the REAL bucket's ECE falls in that
    picture. If it's way out past where random luck usually lands,
    it's a real effect. If it looks like a totally normal random draw,
    it's noise — the bucket's fights just happened to be uneven.

    This directly targets the pattern in the layoff table (smaller n
    -> bigger ECE, almost every time) which is exactly what pure
    sampling noise looks like. This function tells you whether that's
    ALL that's happening, or whether there's a real effect sitting on
    top of it.

    Parameters
    ----------
    table : pd.DataFrame
        Output of bucket_ece() — needs 'bucket', 'n', 'ece' columns.
    y_true, y_prob : np.ndarray
        The SAME full-val arrays bucket_ece() was run on. Random
        draws come from this whole pool, not from within one bucket
        — that's what makes it a fair "what does luck alone produce"
        comparison.
    buckets_to_check : list[str]
        Which bucket names (from table['bucket']) to test. Only test
        the ones that actually matter for the ADR — no need to burn
        time permutation-testing every bucket.
    n_permutations : int
        How many random draws to build the null picture from. 2000
        gives p-value resolution to about 0.0005 — fine enough to
        tell "clearly real" from "borderline" from "just noise."
    seed : int
        Fixed so a rerun reproduces the same verdict.

    Returns
    -------
    pd.DataFrame — one row per bucket:
        bucket, n, observed_ece, null_mean_ece, null_p95_ece,
        percentile, p_value, likely_real

    Reading it:
    - null_mean_ece / null_p95_ece: what ECE typically looks like for
      a RANDOM group of that size, and what a "pretty unlucky but
      still meaningless" random group looks like (95th percentile).
    - percentile: where the real bucket's ECE lands among the 2,000
      random draws. 97th percentile = worse than 97% of random draws
      that size — a real effect. 55th percentile = totally ordinary.
    - p_value: fraction of random draws that were AS BAD OR WORSE than
      the real bucket. Below 0.05 is the usual "probably not luck"
      line — same threshold used everywhere in stats, not special to
      this project.
    - likely_real: p_value < 0.05, as a plain yes/no you can put in
      the ADR without re-deriving it every time you reread this.
    """
    rng = np.random.default_rng(seed)
    n_total = len(y_true)
    rows = []

    for bucket in buckets_to_check:
        match = table[table["bucket"] == bucket]
        if match.empty:
            raise ValueError(f"'{bucket}' not found in table — check spelling.")

        n = int(match["n"].iloc[0])
        observed_ece = float(match["ece"].iloc[0])

        if n > n_total:
            raise ValueError(f"bucket '{bucket}' has n={n} > full val n={n_total}.")

        null_eces = np.empty(n_permutations)
        for i in range(n_permutations):
            idx = rng.choice(n_total, size=n, replace=False)
            null_eces[i] = expected_calibration_error(y_true[idx], y_prob[idx])

        percentile = float((null_eces <= observed_ece).mean() * 100)
        p_value = float((null_eces >= observed_ece).mean())

        rows.append(
            {
                "bucket": bucket,
                "n": n,
                "observed_ece": observed_ece,
                "null_mean_ece": float(null_eces.mean()),
                "null_p95_ece": float(np.percentile(null_eces, 95)),
                "percentile": percentile,
                "p_value": p_value,
                "likely_real": p_value < 0.05,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    train, val = build_train_val_with_elo()
    y_true, y_prob, _, _ = train_lightgbm_baseline(
        train, val, params=load_tuned_params()
    )

    overall = _as_dict(evaluate(y_true, y_prob, name="tuned_full_val"))
    print(f"overall full-val ECE: {overall['ece']:.4f} "
          f"(pre-registered baseline: {BASELINE_ECE})\n")

    fights = bucket_ece(
        val, y_true, y_prob,
        column="self_total_ufc_fights",
        bins=FIGHT_COUNT_BINS,
        labels=FIGHT_COUNT_LABELS,
    )
    print("=== ECE by self_total_ufc_fights ===")
    print(fights.to_string(index=False))

    layoff = bucket_ece(
        val, y_true, y_prob,
        column="self_days_since_last_fight",
        bins=LAYOFF_BINS,
        labels=LAYOFF_LABELS,
    )
    print("\n=== ECE by self_days_since_last_fight ===")
    print(layoff.to_string(index=False))

    # Only the ADR-named populations count toward the pre-registered gate.
    fight_opens, fight_reasons = gate_verdict(
        fights, gate_buckets=["0 (debut)", "1-2", "3-5"]
    )
    layoff_opens, layoff_reasons = gate_verdict(
        layoff, gate_buckets=["365-730d", "730d+", "no prior fight"]
    )

    print("\n=== ADR-014 gate (pre-registered rule) ===")
    for r in fight_reasons + layoff_reasons:
        print(f"  {r}")
    print(
        f"\nVERDICT: {'OPEN — build Glicko-2 RD' if (fight_opens or layoff_opens) else 'CLOSED — Glicko-2 to IDEAS.md'}"
    )

    # --- Is any of that real, or is it small-bucket noise? ---
    # Diagnostic only — does NOT change the verdict above. That
    # verdict was pre-registered before results existed and stays as
    # the record of what the rule said. This just tells you how much
    # to trust it before it goes in ADR-015.
    print("\n=== Permutation check: real effect vs. small-bucket noise ===")
    fight_null = check_buckets_against_null(
        fights, y_true, y_prob,
        buckets_to_check=["0 (debut)", "21+"],
    )
    print(fight_null.to_string(index=False))

    layoff_null = check_buckets_against_null(
        layoff, y_true, y_prob,
        buckets_to_check=["365-730d", "730d+"],
    )
    print(layoff_null.to_string(index=False))