"""
Expanding-window, year-boundary cross-validation for Week 3 Monday's
LightGBM tuning (docs/PLAN.md Section 3, Optuna step). This is evaluation-harness 
/ leakage-defense logic, the same category as features/split.py's outer temporal 
split and features/symmetrize.py's _symmetrize_row, per the project's
agent/human code boundary.

Why this exists on top of the outer train/val/test split (features/
split.py): that split answers "is the FINAL model any good," using
val exactly once per model variant. Optuna needs to ask "is THIS
hyperparameter combination any good" dozens or hundreds of times — if
it asked that question against the real val set, val would stop being
a clean, untouched check and would become just another thing tuned
against, the same trap docs/PLAN.md Section 3 warns about for the
test set ("you get one look"). Expanding-window folds carve train
itself into repeated smaller train/val splits so Optuna gets its many
looks entirely inside train, leaving the real val set completely
unbothered until Step 8.

FOLD BOUNDARIES: hand-picked from real year-by-year bout counts
(checked directly against train.parquet this session), not a generic
sklearn TimeSeriesSplit default. 1999-2010 is bundled into a single
warm-up training block (1999 alone has 19 bouts — nowhere near enough
to validate anything on its own); 2011 onward, each year folds
individually, since by then annual bout counts (295-506/year) are in
the same range as the real val split's ~500/year and large enough to
trust as a validation signal.
"""

from collections.abc import Iterator

import pandas as pd

WARMUP_END_YEAR = 2010  # last year folded into the initial training block, not its own fold


def expanding_year_folds(
    df: pd.DataFrame, warmup_end_year: int = WARMUP_END_YEAR
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yields (train_fold, val_fold) DataFrame pairs, one per calendar
    year from warmup_end_year+1 through the last year present in df.

    Simple version: sort every fight by date, then ask "if I only knew
    everything up through this year, how would I have done on NEXT
    year's fights?" — over and over, moving one year forward each
    time. Fold 1 trains on 1999-2010, validates on 2011. Fold 2 trains
    on 1999-2011, validates on 2012. And so on, through training on
    1999-2021, validating on 2022 (train's last full year per
    features/split.py's boundary).

    Deliberately just a date filter, not a recomputation of any
    feature. Every column already in df (Tier 1/2 stats from
    store.py, self_elo_pre/opp_elo_pre from build_lgbm_matrix.py) was
    already computed as-of each bout's OWN event_date, independent of
    where any CV fold boundary later falls — that's what makes
    "filter by date" sufficient here rather than needing to rebuild
    anything per fold. The leakage-safety work already happened
    upstream; this function's only job is drawing fold lines on top
    of data that's already safe.

    Because event_date is identical for both rows of a symmetrized
    bout_id pair, filtering by date alone naturally keeps a bout's two
    rows together in the same fold — same guarantee
    notebooks/leakage_audit.py's split_integrity_check() verified for
    the outer split, automatic here by construction rather than
    something to check after the fact.

    Parameters
    ----------
    df : pd.DataFrame
        The symmetrized train split, WITH Elo attached — i.e.
        features.build_lgbm_matrix.build_train_val_with_elo()'s first
        return value. Needs event_date and bout_id. Passing the raw
        train.parquet (no Elo) would silently produce folds missing
        diff_elo_pre once to_differential() runs on them; passing val
        or test would silently tune against data meant to stay
        untouched. Neither is checked inside this function.
    warmup_end_year : int
        Last calendar year folded into the initial training block
        rather than becoming its own validation fold. Defaults to
        2010 per the real bout-count check run this session.

    Yields
    ------
    (train_fold, val_fold) : tuple[pd.DataFrame, pd.DataFrame]
        train_fold : every row with event_date's year <= the fold's
            cutoff year.
        val_fold : every row with event_date's year == the fold's
            validation year (exactly one calendar year).

    Raises
    ------
    ValueError if df is empty, or if there's no data at all after
    warmup_end_year (nothing to fold over) — fails loudly rather than
    silently yielding zero folds and letting Step 7's Optuna loop
    quietly do nothing.
    """
    if df.empty:
        raise ValueError("df is empty — nothing to fold over.")

    years = pd.to_datetime(df["event_date"]).dt.year
    val_years = sorted(y for y in years.unique() if y > warmup_end_year)

    if not val_years:
        raise ValueError(
            f"no data found after warmup_end_year={warmup_end_year} — "
            f"check df covers the expected date range."
        )

    for val_year in val_years:
        train_mask = years <= (val_year - 1)
        val_mask = years == val_year

        yield df.loc[train_mask].reset_index(drop=True), df.loc[val_mask].reset_index(drop=True)


if __name__ == "__main__":
    from features.build_lgbm_matrix import build_train_val_with_elo

    train, _ = build_train_val_with_elo()

    print(f"{'val_year':>8} | {'train_bouts':>11} | {'val_bouts':>9} | train date range")
    for train_fold, val_fold in expanding_year_folds(train):
        n_train_bouts = train_fold["bout_id"].nunique()
        n_val_bouts = val_fold["bout_id"].nunique()
        val_year = pd.to_datetime(val_fold["event_date"]).dt.year.iloc[0]
        date_range = (
            f"{train_fold['event_date'].min()} -> {train_fold['event_date'].max()}"
        )
        print(f"{val_year:>8} | {n_train_bouts:>11} | {n_val_bouts:>9} | {date_range}")