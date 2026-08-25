"""
Freeze v1 — Week 3 Saturday (docs/PLAN.md Section 3, ADR-021).

Turns the ADR-020 recipe (Monday's frozen Optuna params, train+val
<=2024) into a physical, hashed, read-only file. Friday's actual
scored model lived in memory for one script run and was never saved
— this file refits that exact recipe and checks, empirically, whether
refitting it gives back the same model. That check matters: it's the
only thing standing between "this file legitimately scored 0.6333 log
loss on test" and a guess.

GUARDRAIL: this file must never read data/test_locked/test.parquet.
It calls build_train_val_with_elo() (train+val only), never
build_all_splits_with_elo() (which reads test.parquet). The test set
is chmod 000 as of Friday's re-lock, so even a mistaken call would
fail loudly with a PermissionError rather than silently succeeding —
but check_test_set_untouched() below confirms the lock is intact
before AND after, as a receipt, not just a hope.
"""

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from features.build_lgbm_matrix import TEST_PARQUET_PATH, build_train_val_with_elo
from features.differential import to_differential
from features.split import TEST_START
from models.lightgbm_model import DEFAULT_PARAMS, load_tuned_params

VERSION = "v1"
MODEL_TYPE = "lightgbm"
RELEASE_DIR = Path("models/v1")
RESULTS_DIR = Path("docs/results")


def check_test_set_untouched(test_path: Path = Path(TEST_PARQUET_PATH)) -> None:
    """
    Confirms test.parquet is still chmod 000 — run before AND after
    the freeze. Doesn't read the file's contents (impossible at this
    permission level); just stats its mode bits, which requires no
    read permission on the file itself.

    Same "receipt, not assumption" instinct as test_eval.py's
    elo_regression_check — the reasoning for why this file can't leak
    is sound, but a cheap direct check costs nothing and catches the
    one scenario the reasoning doesn't cover: someone chmod'd it back
    to readable between Friday and today.
    """
    mode = os.stat(test_path).st_mode & 0o777
    assert mode == 0o000, (
        f"expected {test_path} to be locked at chmod 000; found "
        f"{oct(mode)}. Stop — re-lock it before running a freeze."
    )


def fit_and_check_determinism(
    X: pd.DataFrame, y: pd.Series, model_params: dict, n_checks: int = 2
) -> tuple[lgb.LGBMClassifier, dict]:
    """
    Fits the model n_checks times on identical data and params,
    compares each fit's serialized tree structure byte-for-byte, and
    returns the FIRST fit as the artifact to freeze.

    Why this can fail even with random_state fixed: LightGBM's default
    histogram construction can sum floating-point values in a
    thread-dependent order when using multiple cores, which can nudge
    a split threshold by a tiny amount run to run. A seed alone
    doesn't rule this out — deterministic=True does, but that flag
    isn't set anywhere in this project's params today (confirmed by
    reading lgbm_best_params.json directly, not assumed).

    If the two fits differ, that's still a useful, honestly-reportable
    result — it means the model card says "matches within floating-
    point noise," not "bit-identical." Either outcome is fine; a
    silent, unchecked assumption would not have been.
    """
    fits = []
    for _ in range(n_checks):
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X, y)
        fits.append(model)

    dumps = [m.booster_.model_to_string() for m in fits]
    identical = all(d == dumps[0] for d in dumps[1:])

    report = {
        "bit_identical": identical,
        "n_checks": n_checks,
        "note": (
            "refits produced byte-identical LightGBM text dumps"
            if identical
            else "refits DIFFERED — see docs/MODEL_CARD.md for what this "
            "means for reported metrics. Consider adding "
            "deterministic=True and force_row_wise=True to model "
            "params for future freezes."
        ),
    }
    return fits[0], report


def sha256_of_file(path: Path) -> str:
    """Content hash of the frozen artifact — tamper evidence, and a
    concrete way to answer 'is this really the file I think it is?'
    six months from now."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    """
    Same logic as models/test_eval.py's git_sha() — the commit (plus
    a -DIRTY suffix for uncommitted changes) active when this script
    ran. Duplicated rather than imported: freeze.py is a standalone,
    reusable release step, not a continuation of the one-time test-
    unlock script.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return f"{sha}-DIRTY" if dirty else sha
    except (subprocess.CalledProcessError, OSError):
        return "UNKNOWN"


def library_versions() -> dict:
    """Captured at runtime, not hardcoded, so this stays accurate as
    dependencies drift across future freezes."""
    return {
        pkg: pkg_version(pkg)
        for pkg in ("lightgbm", "numpy", "pandas", "scikit-learn")
    }


def load_test_unlock_results(results_dir: Path = RESULTS_DIR) -> tuple[dict, str]:
    """
    Loads the most recent test_unlock_*.json — the metrics this
    artifact's card and registry row will report. Never recomputed
    here: they were already produced by the one-time, now-
    irreversible test_eval.py run (ADR-020), and reading test.parquet
    again isn't just unnecessary, it's impossible (chmod 000).
    """
    candidates = sorted(results_dir.glob("test_unlock_2*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no test_unlock_*.json found in {results_dir} — "
            f"run `python -m models.test_eval --unlock` first."
        )
    latest = candidates[-1]
    return json.loads(latest.read_text()), str(latest)


def run() -> None:
    """
    The whole day, in order: verify the lock, build train+val, fit
    twice and compare, serialize, hash, write metadata, lock the
    files down, verify the lock one more time.
    """
    print("=== Week 3 Saturday — Freeze v1 (ADR-021) ===\n")

    check_test_set_untouched()
    print("test.parquet confirmed chmod 000 — will not be read.\n")

    print("building train+val (event_date < 2025-01-01, full-history Elo)...")
    train, val = build_train_val_with_elo()
    train_val = pd.concat([train, val], ignore_index=True)
    n_bouts = int(train_val["bout_id"].nunique())
    print(f"  {len(train_val)} rows / {n_bouts} bouts\n")

    X, y = to_differential(train_val, verbose=False)
    tuned_params = load_tuned_params()
    model_params = {**DEFAULT_PARAMS, **tuned_params}

    print("fitting twice to check determinism (~1-2 min)...")
    model, determinism = fit_and_check_determinism(X, y, model_params)
    print(f"  {determinism['note']}\n")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RELEASE_DIR / "model.txt"
    model.booster_.save_model(str(model_path))
    print(f"model saved to {model_path}")

    artifact_hash = sha256_of_file(model_path)
    results, results_path = load_test_unlock_results()

    metadata = {
        "version": VERSION,
        "model_type": MODEL_TYPE,
        "artifact_path": str(model_path),
        "artifact_sha256": artifact_hash,
        # ORDER MATTERS: Week 4's inference code must build its
        # feature matrix in exactly this column order before calling
        # Booster.predict() on a raw array.
        "feature_list": X.columns.tolist(),
        "training_cutoff": str(TEST_START.date()),
        "training_cutoff_note": (
            "exclusive upper bound — train+val includes event_date < "
            "this date (i.e., through 2024-12-31)"
        ),
        "train_row_count": len(train_val),
        "train_bout_count": n_bouts,
        "tuned_hyperparameters": tuned_params,
        "full_model_params": model_params,
        "is_calibrated": False,
        "git_sha": git_sha(),
        "test_eval_git_sha": results.get("git_sha"),
        "test_eval_results_path": results_path,
        "metrics": results.get("metrics"),
        "verdict": results.get("verdict"),
        "determinism_check": determinism,
        "library_versions": library_versions(),
        "trained_at": datetime.now(UTC).isoformat(),
        "notes": (
            f"Test-day code state recorded as {results.get('git_sha')}. "
            "The uncommitted diff active at that time is not "
            "recoverable from git history (commit 9c08a8c, "
            "immediately following, touched only docs/ and "
            "docs/results/, no code). See ADR-021 Decision 5."
        ),
    }

    metadata_path = RELEASE_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"metadata saved to {metadata_path}\n")

    os.chmod(model_path, 0o444)
    os.chmod(metadata_path, 0o444)
    print(f"{model_path} and {metadata_path} set to chmod 444 (read-only)\n")

    check_test_set_untouched()
    print("confirmed test.parquet still untouched. Freeze complete.")


if __name__ == "__main__":
    run()