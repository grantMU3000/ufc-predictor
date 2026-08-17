"""
Turns the symmetrized self_/opp_ training matrix into differential
(self-minus-opp) features for the logistic regression baseline —
Week 2 Thursday, docs/PLAN.md Section 3.

Why differential, not raw self_/opp_ values: a linear model fed raw
self_slpm and opp_slpm separately can still learn "self_slpm's
coefficient is bigger than opp_slpm's coefficient," which is a softer
version of exactly the corner-ordering leak this whole pipeline was
built to defeat (ADR-004). A model fed ONLY self_slpm - opp_slpm
physically cannot express "being self matters more than being opp" —
the two sides are forced to be mirror images by construction. That's
what makes P(self wins) + P(opp wins) = 1 true BY CONSTRUCTION rather
than something you're hoping the model learned, provided it's also
fit with fit_intercept=False (models/baselines.py, Step 6).
"""

import pandas as pd

# Real self_/opp_ pairs that must NEVER be differenced — these aren't
# fighter stats, they're identity/bookkeeping columns that happen to
# share the self_/opp_ naming convention. self_fighter_id - opp_fighter_id
# is a meaningless number (an arbitrary database ID, not a stat), and
# including it would silently hand the model a corner/identity-adjacent
# signal to key off of — exactly what ADR-004 exists to prevent.
_ID_SUFFIXES = {"fighter_id"}


def to_differential(
    df: pd.DataFrame, verbose: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds diff_<feature> = self_<feature> - opp_<feature> for every
    numeric self_/opp_ column pair in a symmetrized dataframe.

    Simple version: for every stat you have on both fighters (reach,
    SLpM, days since last fight), this collapses two numbers into
    one — "how much MORE of this does self have than opp." A positive
    number always means "self has more," no matter which fighter self
    happens to be. That's the whole trick that makes this safe to
    feed a linear model.

    Three categories of column, handled three different ways:
      1. Paired, numeric self_/opp_ columns (e.g. self_slpm/opp_slpm)
         -> diff_slpm = self_slpm - opp_slpm. Included in X.
      2. Paired, but NON-numeric (e.g. self_stance/opp_stance are
         category strings like "orthodox" — you can't subtract two
         words) -> dropped, logged as non_numeric_dropped.
      3. self_-only columns with no opp_ counterpart (e.g.
         self_is_open_stance_matchup — stance_matchup's opp_ side was
         already dropped in symmetrize.py, see that module's
         docstring) -> dropped, logged as self_only_dropped. These
         are facts about the PAIR, identical in both rows of a bout's
         flipped pair, so keeping them as a raw (non-differenced)
         feature would break the "P(self)+P(opp)=1" property the same
         way a bout-level feature (is_title_fight, weight_class)
         would.

    Columns with no self_/opp_ prefix at all (bout_id, event_date,
    weight_class, is_title_fight, scheduled_rounds, source_corner)
    are never touched — they simply don't match the self_/opp_
    pairing logic, so they never enter X. This IS the mechanism, not
    a separate exclude list: a symmetric linear model gets zero
    benefit from a feature identical in both of a bout's two rows, so
    there's nothing to gain by including it here. (Worth noting
    plainly: this means stance information carries zero weight in
    today's LR baseline — not a bug, just what falls out of "only
    per-fighter differences count" for a purely linear model. It'll
    matter again with LightGBM in Week 3, where interaction effects
    CAN use bout-level facts like this — a separate decision for that
    week, not something this function tries to anticipate.)

    Parameters
    ----------
    df : pd.DataFrame
        A symmetrized split (e.g. data/processed/train.parquet or
        val.parquet). Must contain self_won.
    verbose : bool
        Prints a short report of what was included/dropped and why —
        default True, since silently dropping columns is exactly the
        kind of thing worth eyeballing every run, not just the first.

    Returns
    -------
    (X, y)
        X : pd.DataFrame of diff_* columns only, numeric, NaNs
            preserved (NOT imputed here — imputation is
            models/baselines.py's job in Step 6, kept separate on
            purpose so "build the features" and "handle missingness"
            never tangle into one function neither can be tested
            cleanly on its own).
        y : pd.Series of 0/1 ints, self_won.

    Both X and y keep df's original index. Deliberate, not
    incidental: it means a caller can always look up
    df.loc[X.index, ["bout_id", "self_fighter_id"]] later to join
    predictions back to a specific bout — needed in Step 6, when LR's
    accuracy has to be compared against the market on the exact same
    odds-covered subset of rows.
    """
    y = df["self_won"].astype(int)

    diff_data = {}
    included, non_numeric_dropped, self_only_dropped, opp_only_dropped = [], [], [], []

    self_cols = [c for c in df.columns if c.startswith("self_")]
    opp_cols = {c for c in df.columns if c.startswith("opp_")}
    matched_opp_cols = set()

    for col in self_cols:
        suffix = col[len("self_"):]
        if suffix == "won":
            continue  # already pulled out as y, never a feature
        if suffix in _ID_SUFFIXES:
            opp_col = f"opp_{suffix}"
            if opp_col in opp_cols:
                matched_opp_cols.add(opp_col)
            continue  # identity column, never a feature — see module note

        opp_col = f"opp_{suffix}"
        if opp_col not in opp_cols:
            self_only_dropped.append(suffix)
            continue

        matched_opp_cols.add(opp_col)
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[opp_col]):
            diff_data[f"diff_{suffix}"] = df[col].astype(float) - df[opp_col].astype(float)
            included.append(suffix)
        else:
            non_numeric_dropped.append(suffix)

    # Defensive check: any opp_ column that never found a self_ partner
    # at all. Shouldn't happen given how symmetrize.py builds its
    # output, but cheap to check and loud if it's ever wrong.
    unmatched_opp = opp_cols - matched_opp_cols
    for col in unmatched_opp:
        opp_only_dropped.append(col[len("opp_"):])

    X = pd.DataFrame(diff_data, index=df.index)

    if verbose:
        print(f"to_differential: {len(included)} diff_ features built")
        if self_only_dropped:
            print(f"  dropped (self_-only, no opp_ pair, would break symmetry): {self_only_dropped}")
        if non_numeric_dropped:
            print(f"  dropped (non-numeric, can't subtract): {non_numeric_dropped}")
        if opp_only_dropped:
            print(f"  WARNING dropped (opp_-only, no self_ pair — unexpected): {opp_only_dropped}")

    return X, y