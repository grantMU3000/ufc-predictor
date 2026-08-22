"""
Figures for model reporting — starting with Week 3 Wednesday's
reliability diagram (docs/PLAN.md Section 3: "Plot the reliability
diagram — this plot goes in the README").

Kept separate from models/metrics.py deliberately: metrics.py computes
numbers and is imported by scripts that must not need matplotlib.
Plotting is a presentation concern with a heavier dependency, and it's
the one part of the pipeline whose output is judged by eye.

HOW TO READ A RELIABILITY DIAGRAM: the x-axis is what the model
claimed ("I'm 65% sure"), the y-axis is what actually happened (of all
the fights it claimed 65% on, how many did that fighter win). Perfect
calibration is the 45-degree line. Below the diagonal at x>0.5, or above it at x<0.5, means overconfident — the
model claimed more certainty than reality delivered. The opposite pattern
(above the diagonal at x<0.5, below it at x>0.5) means the model isn't
confident enough — real outcomes are more decisive than the stated
probabilities suggest. The histogram underneath matters as much as the curve: a bin holding 12
fights will bounce around the diagonal no matter how good the model
is, so a dramatic-looking wiggle at the edges is usually just a thin
bin, not a real calibration failure.

WHAT THIS FILE PLOTS AND WHY IT INCLUDES A REJECTED MODEL: v1 ships
UNCALIBRATED (ADR-017). Both calibrators failed their pre-registered
gates, for two different reasons — isotonic on symmetry (1.86x the raw
model's own pair deviation, ADR-004), Platt on insufficient ECE
benefit (-0.0022 against a -0.005 bar). Isotonic is plotted alongside
raw anyway, labelled as rejected, because "here is the curve we chose
not to ship, and here is what it did to val" is a far more honest
README figure than showing only the winner. Platt is NOT plotted: it
was rejected on train-internal evidence and never scored against val,
and plotting it would require taking a val read the project didn't
need and didn't take.

PROVENANCE: this file regenerates the probabilities it plots rather
than loading .npy files left behind by an earlier run of
models/calibration.py. Those files were written during a run whose
verdict was later superseded, and a figure whose inputs came from a
stale artifact is a figure nobody can trace. Everything below is
deterministic (seed-fixed LightGBM, an isotonic fit on the saved OOF
frame), so a rerun reproduces the same PNGs exactly.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI / headless runs; write files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DIR = Path("docs/images")
OOF_PATH = Path("data/processed/oof_predictions.parquet")


def reliability_diagram(
    y_true: np.ndarray,
    prob_sets: dict[str, np.ndarray],
    n_bins: int = 10,
    title: str = "Reliability — tuned LightGBM, validation 2023-24",
    filename: str = "reliability_v1.png",
) -> Path:
    """
    Plots one or more probability sets against the perfect-calibration
    diagonal, with a shared prediction histogram beneath.

    Simple version: pass {"raw (shipped)": p_raw, "isotonic
    (rejected)": p_iso} and it draws both curves on the same axes so
    the comparison is a single picture rather than two plots someone
    has to mentally overlay.

    Bin centres are plotted at each bin's MEAN predicted probability
    rather than at the bin's midpoint. That matters for this model
    specifically: it rarely leaves ~0.25-0.75, so several bins are
    populated only near one edge, and plotting at the midpoint would
    put a marker where no prediction actually lives.

    Parameters
    ----------
    y_true : np.ndarray
        0/1 outcomes.
    prob_sets : dict[str, np.ndarray]
        Label -> predicted probabilities. All must align with y_true.
        Insertion order controls legend and draw order.
    n_bins : int
        Equal-width bins across [0, 1]. 10 is the convention and
        matches models/metrics.py's ECE default — keep them consistent
        or the plot and the number will tell slightly different
        stories.
    title, filename : str

    Returns
    -------
    Path to the written PNG.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")

    for label, probs in prob_sets.items():
        probs = np.asarray(probs, dtype=float)
        idx = np.digitize(probs, edges[1:-1])
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            xs.append(probs[mask].mean())
            ys.append(y_true[mask].mean())
        ax.plot(xs, ys, marker="o", lw=1.5, label=label)
        ax_hist.hist(probs, bins=edges, alpha=0.5, label=label)

    ax.set_ylabel("observed win rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax_hist.set_xlabel("predicted probability")
    ax_hist.set_ylabel("count")
    ax_hist.grid(alpha=0.3)

    out = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def ece_by_bin_count(
    y_true: np.ndarray, prob_sets: dict[str, np.ndarray], bin_counts=(5, 10, 15, 20)
) -> pd.DataFrame:
    """
    Recomputes ECE at several bin counts — Step 5a's sensitivity check,
    done here since it's the same inputs the diagram already has.

    Simple version: ECE is not one number, it's one number PER binning
    choice. Ten bins is a convention, not a law. If a conclusion
    ("isotonic is better calibrated") only holds at 10 bins and flips
    at 15, you found a binning artifact rather than a real effect. If
    it holds across all of them, the finding is robust.

    Relevant to ADR-017 specifically: isotonic's holdout ECE advantage
    was the one number that looked genuinely strong before Gate C
    disqualified it, so it's worth knowing whether that advantage was
    even bin-stable on val.

    Returns
    -------
    pd.DataFrame — one row per bin count, one column per label.
    """
    from models.metrics import expected_calibration_error

    rows = []
    for k in bin_counts:
        row = {"n_bins": k}
        for label, probs in prob_sets.items():
            row[label] = expected_calibration_error(
                y_true, np.asarray(probs, dtype=float), n_bins=k
            )
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from features.build_lgbm_matrix import build_train_val_with_elo
    from models.baselines import _load_val_and_odds, _odds_covered_mask
    from models.calibration import fit_isotonic
    from models.lightgbm_model import load_tuned_params, train_lightgbm_baseline

    # --- Raw val probabilities: Monday's shipped model, regenerated ---
    # Not a new val read. These are the exact numbers already recorded
    # in docs/RESULTS.md (0.6224 / 0.6529 / 0.2304 / 0.0235 full val);
    # LightGBM is seed-fixed, so this reproduces rather than re-asks.
    train, val = build_train_val_with_elo()
    y_val, p_raw, _, _ = train_lightgbm_baseline(
        train, val, params=load_tuned_params()
    )

    # --- Isotonic, refit on the full OOF frame, for the rejected curve ---
    # Same fit models/calibration.py performed when isotonic was still
    # the chosen method; that val pass already happened and is on
    # record in ADR-017. Reproducing it costs no additional look.
    oof = pd.read_parquet(OOF_PATH)
    iso = fit_isotonic(oof["p_raw"].to_numpy(), oof["y_true"].to_numpy())
    p_iso = iso.transform(p_raw)

    _, closing = _load_val_and_odds()
    covered = _odds_covered_mask(val, closing).to_numpy()

    prob_sets = {"raw (shipped v1)": p_raw, "isotonic (rejected)": p_iso}

    reliability_diagram(
        y_val,
        prob_sets,
        title="Reliability — tuned LightGBM, full validation (2023-24)",
        filename="reliability_v1_full_val.png",
    )
    reliability_diagram(
        y_val[covered],
        {k: v[covered] for k, v in prob_sets.items()},
        title="Reliability — tuned LightGBM, odds-covered validation subset",
        filename="reliability_v1_odds_covered.png",
    )

    print("\n=== ECE sensitivity to bin count, full val ===")
    print(ece_by_bin_count(y_val, prob_sets).to_string(index=False))
    print("\n=== ECE sensitivity to bin count, odds-covered ===")
    print(
        ece_by_bin_count(
            y_val[covered], {k: v[covered] for k, v in prob_sets.items()}
        ).to_string(index=False)
    )