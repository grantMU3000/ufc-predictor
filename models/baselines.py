"""
Baseline models for Week 2 Thursday (docs/PLAN.md Section 3). Each
baseline function returns (y_true, y_prob) in the same shape, so every
one of them can be scored through the exact same models.metrics.evaluate
call — no baseline gets graded on a curve.

Today's baseline: the market itself. Tomorrow (Elo) and the LR baseline
get added to this same file, so docs/RESULTS.md eventually pulls every
row from one script, one run, one consistent evaluation pass.
"""

import duckdb
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.differential import to_differential
from features.odds import get_closing_lines
from features.elo import compute_elo_ratings, expected_score, k_factor_by_experience
from features.labels import get_completed_decided_bouts
from features.split import TEST_START


def market_baseline(
    val: pd.DataFrame, closing: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Turns the market's de-vigged closing probability into predictions
    for the "always pick the favorite" baseline — docs/PLAN.md Section
    0.1's actual bar to clear, not the naive 60% in the original scope.

    Simple version: for every fight in val, ask "what did the market
    think self_fighter_id's odds of winning were, right before the
    fight?" That number IS the prediction. No model, no features —
    just the closing line, scored the same way everything else will be.

    Join key: bout_id + self_fighter_id <-> bout_id + fighter_id. This
    is exactly why self_fighter_id was added to _symmetrize_row back in
    Wednesday's session — without it, there'd be no clean way to attach
    a per-fighter market probability onto a self_/opp_ row.

    Parameters
    ----------
    val : pd.DataFrame
        The symmetrized validation split (data/processed/val.parquet).
        Needs bout_id, self_fighter_id, self_won.
    closing : pd.DataFrame
        Output of features.odds.get_closing_lines — bout_id, fighter_id,
        market_prob.

    Returns
    -------
    (y_true, y_prob, merged)
        y_true : 0/1 array, whether self_fighter_id actually won
        y_prob : float array, market's de-vigged probability self_fighter_id wins
        merged : the joined DataFrame itself, kept around for debugging
                 and for the coverage check below — NOT every val row
                 survives this join, only ones with odds coverage.

    Note: INNER join, deliberately. A bout with no odds coverage has no
    market opinion to grade — there's no sensible y_prob to invent for
    it, so it's correctly absent from the baseline's n, not silently
    filled with a guess.
    """
    merged = val.merge(
        closing,
        left_on=["bout_id", "self_fighter_id"],
        right_on=["bout_id", "fighter_id"],
        how="inner",
    )

    y_true = merged["self_won"].astype(int).to_numpy()
    y_prob = merged["market_prob"].to_numpy()

    return y_true, y_prob, merged


def _load_val_and_odds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the symmetrized val split and the closing-line odds table
    from local Parquet — the standard "wire up a DuckDB connection to
    the snapshot" pattern from split.py's __main__ block, reused here
    rather than duplicated by hand.

    Returns
    -------
    (val, closing) — ready to pass straight into market_baseline.
    """
    con = duckdb.connect()
    for table in ["odds_snapshots"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    val = pd.read_parquet("data/processed/val.parquet")
    closing = get_closing_lines(con)
    con.close()

    return val, closing

def build_logreg_pipeline() -> Pipeline:
    """
    The LR baseline's full preprocessing + model chain, in one object
    so train and val always get treated identically.

    Three steps, three different jobs:
      1. SimpleImputer(strategy="median), add_indicator=True - the
         feature store returns real NaN for undefined values on
         purpose (a debutant has no prior-fight stats to compute a
         rate from). LightGBM eats NaN natively; sklearn's
         LogisticRegression does not, so it needs a real number here.
         Median, not mean, since these are skewed rate/count stats
         where a few outlier fighters (extreme layoffs, tiny sample
         sizes) would drag a mean off-center. add_indicator=True keeps
         "this was missing" as its own 0/1 column instead of silently 
         hiding it - a debutant flag is itself real signal, not noise 
         to erase by filling in a plausible-looking number.
      2. StandardScaler() - matters for LR in a way it never will for
         LightGBM. reach_diff lives in centimeters, slpm_diff in 
         strikes-per-minute; unscaled, LR's regularization punishes
         small-magnitude features just for being small-magnitude, not 
         because they're actually weak signal.
      3. LogisticRegression(fit_intercept=False) - the deliberate
         choice from Step 5's docstring. With PURE diff_ features (no
         intercept to break the symmetry), P(self wins) is
         mathematically guaranteed to equal 1 - P(opp wins) for every
         bout's flipped pair. That's not something to hope the model
         learned - it's structural, checked directly in __main__ below.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(fit_intercept=False, max_iter=1000)),
    ])

def logistic_regression_baseline(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """
    Fits the LR baseline on train, scores it on val.

    Imputer and scaler are fit on TRAIN ONLY (via pipeline.fit, called
    once on X_train) then applied to val with .predict_proba - never
    refit on val. Fitting on the combined set would mean val's own
    distribution leaks into how train gets centered/scaled, which is
    the same temporal-leakage family docs/PLAN.md Section 0.2 already
    warns about, just at the preprocessing layer instead of the
    feature layer.

    Parameters
    ----------
    train, val : pd.DataFrame
        The symmetrized splits (data/processed/{train,val}.parquet).

    Returns
    -------
    (y_true, y_prob, pipeline) — same shape as market_baseline's
    return, so both feed models.metrics.evaluate identically. pipeline
    is returned too, for top_features() below.
    """
    X_train, y_train = to_differential(train, verbose=False)
    X_val, y_val = to_differential(val, verbose=False)

    pipeline = build_logreg_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_val)[:, 1]
    return y_val.to_numpy(), y_prob, pipeline

def top_features(pipeline: Pipeline, feature_names: list, n: int = 10) -> pd.DataFrame:
    """
    Ranks fitted LR coefficients by |magnitude| - a quick
    interprability check and a leakage smell-test in one. If a
    missingness INDICATOR column (e.g. "missingIndicator_diff_slpm")
    outranks the real stats, or if some feature you can't explain
    domain-wise dominates the list, that's worth investigating before
    trusting the number, the same instinct behind the plan's "any 3+
    point jump is a leak until proven otherwise" rule.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted pipeline from logistic_regression_baseline.
    feature_names : list
        Column names of X BEFORE the pipeline (i.e. X_train.columns) —
        the imputer expands this list (adds indicator columns), so raw
        column names alone won't line up with pipeline.coef_ without
        this step.
    Returns
    -------
    pd.DataFrame: feature, coef, abs_coef — top n by abs_coef, sorted descending.
    """
    all_names = pipeline[:-1].get_feature_names_out(feature_names)
    coefs = pipeline.named_steps["clf"].coef_[0]
    result = pd.DataFrame({"feature": all_names, "coef": coefs})
    result["abs_coef"] = result["coef"].abs()
    return result.sort_values("abs_coef", ascending=False).head(n).reset_index(drop=True)

def _odds_covered_mask(df: pd.DataFrame, closing: pd.DataFrame) -> pd.Series:
    """
    Boolean mask (same index/order as df) marking which rows have a
    market_prob available. Used to restrict the LR baseline to the 
    SAME odds-covered subset the market baseline was scored on
    (Step 4's 1870 rows), so the two accuracy numbers are being
    compared on identical footing - comparing LR's full-2,034-row
    accuracy against the market's 1,870-row accuracy would be
    comparing two different denominators and drawing a conclusion
    neither number actually supports.
    """
    covered_keys = set(zip(closing["bout_id"], closing["fighter_id"]))
    row_keys = list(zip(df["bout_id"], df["self_fighter_id"]))
    return pd.Series([k in covered_keys for k in row_keys], index=df.index)

def _load_labels_and_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train+val-era bout history (compute_elo_ratings' input) and
    the symmetrized val split (what gets scored) — the two things
    elo_baseline needs.

    Filtered to event_date < TEST_START HERE, not inside
    compute_elo_ratings — same rule as features/elo.py's own module
    docstring: the ratings function doesn't decide what it's allowed
    to see, the caller does, every single time it's called.
    """
    con = duckdb.connect()
    for table in ["fighters", "events", "bouts"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    labels = get_completed_decided_bouts(con)
    labels = labels[pd.to_datetime(labels["event_date"]) < TEST_START].reset_index(drop=True)

    val = pd.read_parquet("data/processed/val.parquet")
    con.close()

    return labels, val

def elo_baseline(
    labels: pd.DataFrame,
    val: pd.DataFrame,
    k_new: float = 80.0,
    k_veteran: float = 24.0,
    decay_scale: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Turns each fighter's pre-fight Elo rating into a win probability
    for val — the "does fight-history-derived skill alone predict
    this outcome" baseline, no other features involved.

    Defaults are today's winning combo from the (k_new, k_veteran,
    decay_scale) grid search — Pareto-optimal against everything else
    tested, log loss 0.0076 better than flat K=32 at a small,
    within-target calibration cost. Left as parameters, not
    hardcoded, so a future re-tune doesn't require editing this
    function's body.

    Simple version: same idea as market_baseline, just with a
    different source of "belief." The market's belief comes from
    what bettors are willing to wager; Elo's belief comes from who's
    beaten whom, weighted by how much history each fighter has.
    Both get converted to a probability and handed to the exact same
    models.metrics.evaluate() call — no baseline is scored any
    differently than any other.

    Parameters
    ----------
    labels : pd.DataFrame
        Train+val-era decided bouts, sorted oldest-first — output of
        features.labels.get_completed_decided_bouts, already filtered
        to event_date < TEST_START by the caller (_load_labels_and_val
        does this). Feeds compute_elo_ratings.
    val : pd.DataFrame
        The symmetrized val split. Needs bout_id, self_fighter_id,
        self_won, source_corner.
    k_new, k_veteran, decay_scale : float
        Passed straight through to k_factor_by_experience.

    Returns
    -------
    (y_true, y_prob, merged) — same shape as market_baseline and
    logistic_regression_baseline's returns, so all three plug into
    models.metrics.evaluate() identically. merged kept around for
    debugging, same reason market_baseline keeps its merged frame.
    """
    def k_fn(fight_count: int) -> float:
        return k_factor_by_experience(
            fight_count, k_new=k_new, k_veteran=k_veteran, decay_scale=decay_scale
        )

    elo = compute_elo_ratings(labels, k_factor=k_fn)

    merged = val.merge(elo, on="bout_id", how="inner")
    is_red = merged["source_corner"] == "red"
    self_elo = np.where(is_red, merged["red_elo_pre"], merged["blue_elo_pre"])
    opp_elo = np.where(is_red, merged["blue_elo_pre"], merged["red_elo_pre"])

    y_true = merged["self_won"].astype(int).to_numpy()
    y_prob = expected_score(self_elo, opp_elo)

    return y_true, y_prob, merged

if __name__ == "__main__":
    from models.metrics import evaluate

    train = pd.read_parquet("data/processed/train.parquet")
    val, closing = _load_val_and_odds()

    # --- Market baseline (Step 4, reprinted here for a side-by-side view) ---
    y_true_mkt, y_prob_mkt, _ = market_baseline(val, closing)
    print(evaluate(y_true_mkt, y_prob_mkt, name="market_baseline"))

    # --- LR baseline, full val ---
    y_val, y_prob_lr, pipeline = logistic_regression_baseline(train, val)
    print(evaluate(y_val, y_prob_lr, name="logreg_full_val"))

    # --- LR baseline, odds-covered subset only (apples-to-apples vs. market) ---
    covered_mask = _odds_covered_mask(val, closing).to_numpy()
    print(evaluate(y_val[covered_mask], y_prob_lr[covered_mask], name="logreg_odds_covered_subset"))

    # --- Structural symmetry check ---
    # With pure diff_ features + fit_intercept=False, every bout's two
    # flipped rows should predict probabilities summing to ~1.0. This
    # SHOULD be true by construction — checking it directly rather
    # than assuming the math worked.
    pair_sums = (
        pd.DataFrame({"bout_id": val["bout_id"].to_numpy(), "y_prob": y_prob_lr})
        .groupby("bout_id")["y_prob"]
        .sum()
    )
    max_deviation = (pair_sums - 1.0).abs().max()
    print(f"max deviation from symmetric pair sum (expect ~0): {max_deviation:.8f}")

    # --- Interpretability / leakage smell-test ---
    X_train_cols = to_differential(train, verbose=False)[0].columns.tolist()
    print(top_features(pipeline, X_train_cols))

    # --- Elo baseline, full val ---
    labels, val_for_elo = _load_labels_and_val()
    y_true_elo, y_prob_elo, elo_merged = elo_baseline(labels, val_for_elo)
    print(evaluate(y_true_elo, y_prob_elo, name="elo_full_val"))

    # --- Elo baseline, odds-covered subset (apples-to-apples vs. market) ---
    elo_covered_mask = _odds_covered_mask(elo_merged, closing).to_numpy()
    print(evaluate(y_true_elo[elo_covered_mask], y_prob_elo[elo_covered_mask], name="elo_odds_covered"))