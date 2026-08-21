"""Run one temporal fold of a feature ablation in an isolated process.

The full five-fold loop holds the modelling frame plus per-fold copies for the
whole run, which exceeds this machine's Windows commit limit. Scoring one fold
per process keeps peak commit low and returns every allocation to the OS
between folds. Results append to a CSV so the folds can be combined afterwards.

Usage:
    python scripts/run_ablation.py --config <yaml> --fold 0 --label no_anchored
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splitting import load_folds  # noqa: E402
from src.models.training import train_cv  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import REPORTS_DIR  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import _load_prepared  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one fold of an ablation.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default="lightgbm")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.train.seed)

    modelling, _holdout, _meta = _load_prepared()
    folds = load_folds("temporal")
    one = [folds[args.fold]]

    result = train_cv(
        modelling,
        one,
        args.model,
        seed=config.train.seed,
        n_jobs=config.train.n_jobs,
        exclude_feature_suffixes=config.features.exclude_feature_suffixes,
    )
    fold_result = result.fold_results[0]
    m = fold_result.metrics

    out = REPORTS_DIR / f"ablation_{args.label}.csv"
    is_new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["label", "model", "fold", "pr_auc", "roc_auc", "precision", "recall", "f1", "n_features"])
        writer.writerow([
            args.label, args.model, args.fold,
            f"{m.pr_auc:.6f}", f"{m.roc_auc:.6f}",
            f"{m.precision:.6f}", f"{m.recall:.6f}", f"{m.f1:.6f}",
            fold_result.n_features,
        ])
    print(f"fold {args.fold}: PR-AUC {m.pr_auc:.4f} ROC-AUC {m.roc_auc:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
