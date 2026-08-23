"""
The one-time test-set evaluation — Week 3 Friday (docs/PLAN.md
Section 3), under the protocol pre-registered in ADR-020.

This file is different from every other model file in the repo in one
respect: it can only produce a trustworthy number ONCE. Every gate,
guard, and check below exists because a mistake discovered after the
run cannot be undone by re-running.

WHAT IT DOES
  1. Verifies extending Elo through 2025 didn't change a single
     train/val rating (halts if it did).
  2. Verifies split integrity — no bout in two splits, both
     symmetrized rows together, every test date >= TEST_START.
  3. Trains TWO artifacts on Monday's frozen hyperparameters:
       B (ships)     — train+val, <=2024, ADR-020 Decision 1
       A (diagnostic)— train only, <=2022
  4. Scores both on test: full, odds-covered, and close-fight slices.
  5. Runs the ADR-020 Decision 4 backtest sweep through
     models/backtest.py (already unit-tested in tests/test_backtest.py).
  6. Writes everything, plus git SHA and timestamp, to
     docs/results/test_unlock_<timestamp>.json and drops a lock file.

DRY-RUN FIRST. `--dry-run` runs the entire pipeline against VAL
instead of test and asserts it reproduces docs/RESULTS.md's published
numbers to 4dp. If the harness has a bug, it surfaces there — on a
split that can be read a thousand more times — not on the one read
that counts.

ADR-020 Decision 1 in code: BOTH artifacts are scored, but `SHIPPING
_ARTIFACT` is fixed at "B" as a module constant. Nothing in this file
compares their scores and picks a winner. That is the entire point.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from features.build_lgbm_matrix import (
    _load_labels_and_elo,
    build_all_splits_with_elo,
    build_train_val_with_elo,
)
from features.odds import get_closing_lines, get_closing_prices
from features.split import TEST_START
from models.backtest import (
    EDGE_THRESHOLDS,
    kelly_simulation,
    run_backtest_sweep,
    select_bets,
)
from models.calibration import max_pair_deviation
from models.lightgbm_model import load_tuned_params, train_lightgbm_baseline
from models.metrics import evaluate, reliability_curve

# --- ADR-020 pre-registered constants -------------------------------

# Decision 1: B ships regardless of which artifact scores better.
SHIPPING_ARTIFACT = "B"

# Decision 3: "close" is the MARKET's opinion, not the model's.
CLOSE_FIGHT_RANGE = (0.40, 0.60)
CLOSE_FIGHT_MIN_N = 150  # below this, report as directional only (ADR-015)

# Decision 5: primary success criterion — did the val estimate hold?
VAL_LOGLOSS_ODDS_COVERED = 0.6483
GENERALIZATION_TOLERANCE = 0.01
ECE_TARGET = 0.05

# Published val numbers the dry-run must reproduce exactly
# (docs/RESULTS.md, Week 3 Monday, tuned LightGBM, odds-covered).
DRY_RUN_EXPECTED = {"accuracy": 0.6305, "log_loss": 0.6483, "ece": 0.0318}

LOCK_PATH = Path("models/artifacts/test_unlock.lock")
RESULTS_DIR = Path("docs/results")


# --- Guards ---------------------------------------------------------


def git_sha() -> str:
    """
    The commit this run executed at — the fingerprint that ties a test
    number to the exact code that produced it.

    Returns "UNKNOWN-DIRTY" rather than raising if git isn't available
    or the tree has uncommitted changes. A dirty tree doesn't block the
    run (that would be its own kind of trap on unlock day), but it does
    get recorded honestly in the results file so nobody later assumes
    the SHA tells the whole story.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return f"{sha}-DIRTY" if dirty else sha
    except (subprocess.CalledProcessError, OSError):
        return "UNKNOWN"


def check_lock(force: bool) -> None:
    """
    Refuses to run if a previous unlock already happened.

    Simple version: the sealed envelope only opens once. If the lock
    file exists, this run has already been done, and re-running would
    quietly turn the test set into a second validation set — the exact
    failure docs/PLAN.md Section 3 warns about ("if you tune against
    it, you've lost your only honest estimate").

    --force exists for the one legitimate case: the run crashed
    mid-way and produced no usable numbers. Using it after a
    successful run is a protocol violation, and the override is
    recorded in the results file so it can't happen silently.
    """
    if LOCK_PATH.exists() and not force:
        prior = json.loads(LOCK_PATH.read_text())
        raise SystemExit(
            f"\nTest set already unlocked at {prior['timestamp']} "
            f"(git {prior['git_sha']}).\n"
            f"Results: {prior['results_path']}\n\n"
            f"Re-running would make test a second validation set. "
            f"If the prior run genuinely failed and produced nothing "
            f"usable, pass --force — and record why in ADR-020.\n"
        )


def elo_regression_check(tolerance: float = 0.0) -> dict:
    """
    Proves that extending Elo through the test era changed nothing
    about train/val ratings.

    Simple version: Elo is a running tally. Each bout's pre-fight
    rating is written down BEFORE that bout's result is applied
    (features/elo.py), so adding more fights to the END of the walk
    can't reach backward and alter an earlier one. That's the
    reasoning — this function is the receipt.

    It matters because ADR-020 Decision 2 extends the Elo cutoff for
    the first time in the project. If this check ever failed, it would
    mean either the Elo walk isn't order-independent the way the
    docstring claims, or the bout population itself shifted — and
    either way, every val number in docs/RESULTS.md would be
    describing a different model than the one about to be scored.

    tolerance=0.0 deliberately: byte-identical, not "close enough."
    There is no mechanism by which these should differ by 1e-12.

    Returns
    -------
    dict — n_compared, max_abs_diff, passed. Raises on failure.
    """
    _, elo_bounded = _load_labels_and_elo(cutoff=TEST_START)
    _, elo_full = _load_labels_and_elo(cutoff=None)

    merged = elo_bounded.merge(
        elo_full, on="bout_id", how="left", suffixes=("_bounded", "_full")
    )
    assert merged["red_elo_pre_full"].notna().all(), (
        "a pre-TEST_START bout vanished when the Elo cutoff was extended — "
        "the bout population itself changed, not just its boundary."
    )

    red_diff = (merged["red_elo_pre_bounded"] - merged["red_elo_pre_full"]).abs()
    blue_diff = (merged["blue_elo_pre_bounded"] - merged["blue_elo_pre_full"]).abs()
    max_diff = float(max(red_diff.max(), blue_diff.max()))

    if max_diff > tolerance:
        raise SystemExit(
            f"\nHALTED BEFORE UNLOCK: extending the Elo cutoff changed "
            f"{int((red_diff > tolerance).sum() + (blue_diff > tolerance).sum())} "
            f"train/val ratings (max diff {max_diff:.10f}).\n"
            f"Every published val number assumes these are identical. "
            f"Investigate before reading the test set.\n"
        )

    return {"n_compared": len(merged), "max_abs_diff": max_diff, "passed": True}


def split_integrity_check(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """
    The Week 2 split checks, rerun with test included — the follow-up
    LEAKAGE_LOG.md explicitly deferred to this day ("rerun including
    test once Week 3 Friday's test-set unlock happens").

    Four things, each a different way contamination could hide:
      1. No bout_id appears in more than one split.
      2. Every test bout has exactly 2 rows (its symmetrized pair) —
         a bout split across the boundary would mean one perspective
         got graded and the other didn't.
      3. Every test event_date >= TEST_START, and val's max is below
         it — the temporal wall is intact.
      4. self_won is 50/50 within test, guaranteed by symmetrization.

    Returns
    -------
    dict of findings, ready to paste into LEAKAGE_LOG.md. Raises on
    any violation.
    """
    ids = {
        "train": set(train["bout_id"]),
        "val": set(val["bout_id"]),
        "test": set(test["bout_id"]),
    }
    overlaps = {
        f"{a}&{b}": len(ids[a] & ids[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    assert all(v == 0 for v in overlaps.values()), f"split overlap: {overlaps}"

    row_counts = test.groupby("bout_id").size()
    bad_pairs = row_counts[row_counts != 2]
    assert bad_pairs.empty, (
        f"{len(bad_pairs)} test bouts don't have exactly 2 rows: "
        f"{bad_pairs.head().to_dict()}"
    )

    test_dates = pd.to_datetime(test["event_date"])
    val_dates = pd.to_datetime(val["event_date"])
    assert test_dates.min() >= TEST_START, "a test row predates TEST_START"
    assert val_dates.max() < TEST_START, "a val row is inside the test window"

    counts = test["self_won"].value_counts()
    assert counts.get(True, 0) == counts.get(False, 0), (
        f"test not balanced: {counts.to_dict()}"
    )

    return {
        "overlaps": overlaps,
        "n_test_bouts": int(test["bout_id"].nunique()),
        "n_test_rows": len(test),
        "test_date_min": str(test_dates.min().date()),
        "test_date_max": str(test_dates.max().date()),
        "passed": True,
    }


# --- Evaluation -----------------------------------------------------


def _load_odds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Both odds views in one connection: de-vigged belief (for edge) and
    raw decimal price (for money). ADR-020 Decision 4 keeps these
    strictly separate — see features/odds.get_closing_prices.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE VIEW odds_snapshots AS "
        "SELECT * FROM read_parquet('data/processed/odds_snapshots.parquet')"
    )
    lines = get_closing_lines(con)
    prices = get_closing_prices(con)
    con.close()
    return lines, prices


def build_eval_frame(
    split: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray
) -> pd.DataFrame:
    """
    One row per symmetrized row, carrying everything downstream needs:
    outcome, model probability, market belief, market price.

    LEFT join on odds, not inner — unlike models/baselines.py's
    market_baseline, which inner-joins because a bout with no odds has
    no market opinion to grade. Here the model DOES have an opinion on
    every bout, so full-test metrics should include uncovered bouts;
    the odds-covered slice is then taken as a mask on this frame rather
    than as a separately-joined dataset. Same denominator discipline
    as _odds_covered_mask, just organized around one frame.

    Returns
    -------
    pd.DataFrame — bout_id, event_date, self_fighter_id, self_won,
    model_prob, market_prob, decimal_odds, is_odds_covered, is_close.
    """
    lines, prices = _load_odds()

    frame = pd.DataFrame(
        {
            "bout_id": split["bout_id"].to_numpy(),
            "event_date": pd.to_datetime(split["event_date"].to_numpy()),
            "self_fighter_id": split["self_fighter_id"].to_numpy(),
            "self_won": y_true.astype(bool),
            "model_prob": y_prob,
        }
    )

    frame = frame.merge(
        lines.rename(columns={"fighter_id": "self_fighter_id"}),
        on=["bout_id", "self_fighter_id"],
        how="left",
    )
    frame = frame.merge(
        prices.rename(columns={"fighter_id": "self_fighter_id"}),
        on=["bout_id", "self_fighter_id"],
        how="left",
    )

    frame["is_odds_covered"] = frame["market_prob"].notna() & frame["decimal_odds"].notna()

    lo, hi = CLOSE_FIGHT_RANGE
    frame["is_close"] = (
        frame["is_odds_covered"]
        & frame["market_prob"].between(lo, hi, inclusive="both")
    )
    return frame.sort_values("event_date").reset_index(drop=True)


def score_slices(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Runs models.metrics.evaluate over the three ADR-020 populations,
    plus the market's own numbers on the covered slices for a direct
    side-by-side.

    Three slices, three different questions:
      full            — how good is the model on every 2025+ fight?
      odds_covered    — how good is it where the market also has an
                        opinion? (the only fair comparison population)
      close           — how good is it on the fights the MARKET calls
                        a coinflip? (Decision 3 — market-defined, not
                        model-defined, to avoid circularity)

    The market rows use market_prob as if it were a prediction —
    exactly what models/baselines.market_baseline does, scored through
    the same evaluate() call, so nothing is graded on a curve.
    """
    rows = []

    def _add(mask: pd.Series, slice_name: str, prob_col: str, who: str) -> None:
        sub = frame[mask]
        if sub.empty:
            return
        result = evaluate(
            sub["self_won"].astype(int).to_numpy(),
            sub[prob_col].to_numpy(),
            name=f"{who}_{slice_name}",
        )
        result.update(
            {
                "artifact": label,
                "slice": slice_name,
                "who": who,
                "max_pair_dev": max_pair_deviation(
                    sub["bout_id"].to_numpy(), sub[prob_col].to_numpy()
                ),
                "n_bouts": int(sub["bout_id"].nunique()),
            }
        )
        rows.append(result)

    all_rows = pd.Series(True, index=frame.index)
    _add(all_rows, "full", "model_prob", "model")
    _add(frame["is_odds_covered"], "odds_covered", "model_prob", "model")
    _add(frame["is_odds_covered"], "odds_covered", "market_prob", "market")
    _add(frame["is_close"], "close", "model_prob", "model")
    _add(frame["is_close"], "close", "market_prob", "market")

    return pd.DataFrame(rows)


def bootstrap_logloss_ci(
    frame: pd.DataFrame, prob_col: str = "model_prob", n_boot: int = 2000,
    seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Percentile bootstrap CI on log loss — resampled by BOUT, not by row.

    Resampling by bout matters here in a way it wouldn't for an
    unsymmetrized dataset: a bout's two rows are the same fight seen
    from both corners, so their errors are near-perfectly dependent.
    Resampling rows independently would treat them as two pieces of
    evidence instead of one and produce an interval roughly sqrt(2)
    too narrow — a confidently wrong error bar, which is worse than
    none.

    Same purpose as models/backtest.bootstrap_roi_ci and ADR-015's
    permutation test: establish what the noise looks like before
    reading anything into the point estimate.
    """
    grouped = {
        bid: sub for bid, sub in frame.groupby("bout_id")[["self_won", prob_col]]
    }
    bout_ids = np.array(list(grouped.keys()))
    rng = np.random.default_rng(seed)

    losses = []
    for _ in range(n_boot):
        picked = rng.choice(bout_ids, size=len(bout_ids), replace=True)
        sample = pd.concat([grouped[b] for b in picked], ignore_index=True)
        y = sample["self_won"].astype(int).to_numpy()
        p = np.clip(sample[prob_col].to_numpy(), 1e-15, 1 - 1e-15)
        losses.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return (
        float(np.percentile(losses, 100 * alpha / 2)),
        float(np.percentile(losses, 100 * (1 - alpha / 2))),
    )


def interpret(metrics: pd.DataFrame) -> dict:
    """
    Applies ADR-020 Decision 5's success criteria mechanically, so the
    verdict is read off the pre-registered rules rather than narrated
    after the fact.

    Primary: did test log loss (shipping artifact, odds-covered) land
    within GENERALIZATION_TOLERANCE of val's 0.6483? That's the actual
    scientific claim — "my validation estimate generalized" — and
    landing where predicted is a better result than an unexplained
    improvement.

    Secondary: ECE <= 0.05.

    Explicitly recorded as NOT criteria: beating the market, positive
    ROI. Both are reported; neither decides anything.
    """
    ship = metrics[
        (metrics["artifact"] == SHIPPING_ARTIFACT)
        & (metrics["slice"] == "odds_covered")
    ]
    model = ship[ship["who"] == "model"].iloc[0]
    market = ship[ship["who"] == "market"].iloc[0]

    drift = model["log_loss"] - VAL_LOGLOSS_ODDS_COVERED

    return {
        "shipping_artifact": SHIPPING_ARTIFACT,
        "test_log_loss_odds_covered": float(model["log_loss"]),
        "val_log_loss_odds_covered": VAL_LOGLOSS_ODDS_COVERED,
        "drift_from_val": float(drift),
        "primary_criterion_met": bool(abs(drift) <= GENERALIZATION_TOLERANCE),
        "test_ece": float(model["ece"]),
        "secondary_criterion_met": bool(model["ece"] <= ECE_TARGET),
        "market_log_loss": float(market["log_loss"]),
        "gap_to_market": float(model["log_loss"] - market["log_loss"]),
        "beat_market": bool(model["log_loss"] < market["log_loss"]),
        "note": (
            "beat_market and ROI are reported, not success criteria "
            "(ADR-020 Decision 5)."
        ),
    }


# --- Orchestration --------------------------------------------------


def run(dry_run: bool, force: bool) -> None:
    """
    The whole day, in order. Dry-run and real run share every line of
    logic below except which split gets scored — which is the point.
    A harness validated on val is a harness you can trust on test.
    """
    params = load_tuned_params()
    started = datetime.now(UTC)
    sha = git_sha()

    if dry_run:
        print("=== DRY RUN — scoring VAL, test set untouched ===\n")
        train, val = build_train_val_with_elo()
        y_true, y_prob, _, _ = train_lightgbm_baseline(train, val, params=params)
        frame = build_eval_frame(val, y_true, y_prob)
        metrics = score_slices(frame, label="A")

        covered = metrics[
            (metrics["slice"] == "odds_covered") & (metrics["who"] == "model")
        ].iloc[0]
        print(metrics.to_string(index=False))

        print("\n--- reproduction check vs docs/RESULTS.md ---")
        ok = True
        for key, expected in DRY_RUN_EXPECTED.items():
            actual = float(covered[key])
            match = abs(actual - expected) < 5e-5
            ok &= match
            print(f"  {key:10s} expected {expected:.4f}  got {actual:.4f}  "
                  f"{'OK' if match else 'MISMATCH'}")
        if not ok:
            raise SystemExit(
                "\nDRY RUN FAILED — the harness does not reproduce published "
                "val numbers. Do NOT unlock until this is resolved.\n"
            )

        sweep = run_backtest_sweep(
            frame[frame["is_odds_covered"]], label="A_val_dryrun"
        )
        print("\n--- backtest sweep (val, sanity only) ---")
        print(sweep.to_string(index=False))
        print("\nDry run passed. Harness reproduces val exactly.")
        return

    # --- THE UNLOCK ---
    check_lock(force)
    print("=== TEST SET UNLOCK — one-time read (ADR-020) ===\n")

    print("preflight 1/2: Elo regression check...")
    elo_check = elo_regression_check()
    print(f"  {elo_check['n_compared']} bouts compared, "
          f"max diff {elo_check['max_abs_diff']:.2e} — OK\n")

    train, val, test = build_all_splits_with_elo()

    print("preflight 2/2: split integrity (train/val/test)...")
    split_check = split_integrity_check(train, val, test)
    print(f"  {split_check['n_test_bouts']} test bouts, "
          f"{split_check['test_date_min']} to {split_check['test_date_max']} — OK\n")

    train_val = pd.concat([train, val], ignore_index=True)

    artifacts = {
        "B": ("train+val (<=2024) — SHIPS", train_val),
        "A": ("train only (<=2022) — diagnostic", train),
    }

    all_metrics, all_sweeps, frames = [], [], {}
    for name, (desc, fit_data) in artifacts.items():
        print(f"training artifact {name}: {desc} ({len(fit_data)} rows)...")
        y_true, y_prob, _, _ = train_lightgbm_baseline(fit_data, test, params=params)
        frame = build_eval_frame(test, y_true, y_prob)
        frames[name] = frame
        all_metrics.append(score_slices(frame, label=name))
        all_sweeps.append(
            run_backtest_sweep(frame[frame["is_odds_covered"]], label=name)
        )

    metrics = pd.concat(all_metrics, ignore_index=True)
    sweep = pd.concat(all_sweeps, ignore_index=True)

    ship_frame = frames[SHIPPING_ARTIFACT]
    ll_lo, ll_hi = bootstrap_logloss_ci(ship_frame[ship_frame["is_odds_covered"]])
    verdict = interpret(metrics)
    verdict["log_loss_ci_95"] = [ll_lo, ll_hi]

    close_n = int(metrics[
        (metrics["artifact"] == SHIPPING_ARTIFACT) & (metrics["slice"] == "close")
    ]["n"].iloc[0]) if (metrics["slice"] == "close").any() else 0
    verdict["close_fight_n"] = close_n
    verdict["close_fight_directional_only"] = close_n < CLOSE_FIGHT_MIN_N

    print("\n=== METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== BACKTEST SWEEP (ADR-020 Decision 4) ===")
    print(sweep.to_string(index=False))
    print("\n=== VERDICT (ADR-020 Decision 5) ===")
    print(json.dumps(verdict, indent=2))

    # --- Persist everything ---
    stamp = started.strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"test_unlock_{stamp}.json"

    payload = {
        "timestamp": started.isoformat(),
        "git_sha": sha,
        "forced_rerun": force,
        "adr": "ADR-020",
        "shipping_artifact": SHIPPING_ARTIFACT,
        "tuned_params": params,
        "preflight": {"elo": elo_check, "split": split_check},
        "metrics": metrics.to_dict(orient="records"),
        "backtest": sweep.to_dict(orient="records"),
        "verdict": verdict,
    }
    results_path.write_text(json.dumps(payload, indent=2, default=str))

    metrics.to_csv(RESULTS_DIR / f"test_unlock_metrics_{stamp}.csv", index=False)
    sweep.to_csv(RESULTS_DIR / f"test_unlock_backtest_{stamp}.csv", index=False)

    # Reliability curve for the README plot + the Kelly equity curve.
    ship_covered = ship_frame[ship_frame["is_odds_covered"]]
    reliability_curve(
        ship_covered["self_won"].astype(int).to_numpy(),
        ship_covered["model_prob"].to_numpy(),
    ).to_csv(RESULTS_DIR / f"test_reliability_{stamp}.csv", index=False)

    _, ledger = kelly_simulation(select_bets(ship_covered, EDGE_THRESHOLDS[-1]))
    if not ledger.empty:
        ledger.to_csv(RESULTS_DIR / f"test_kelly_ledger_{stamp}.csv", index=False)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps({
        "timestamp": started.isoformat(),
        "git_sha": sha,
        "results_path": str(results_path),
    }, indent=2))

    print(f"\nwrote {results_path}")
    print(f"locked at {LOCK_PATH} — this run cannot be repeated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3 Friday test-set evaluation")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="score val instead of test; verifies the harness reproduces "
             "published val numbers. Run this FIRST.",
    )
    parser.add_argument(
        "--unlock", action="store_true",
        help="perform the one-time test-set read. Required; the flag exists "
             "so this cannot happen from muscle memory.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="override the lock file. Only for a prior run that crashed "
             "without producing usable numbers.",
    )
    args = parser.parse_args()

    if not (args.dry_run or args.unlock):
        raise SystemExit("pass --dry-run first, then --unlock.")
    run(dry_run=args.dry_run, force=args.force)