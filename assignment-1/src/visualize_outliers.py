"""
Visualize outliers: before/after histograms, boxplots with IQR fences, and a
summary of percent capped per column.

Usage (from repo root):
    python assignment-1/src/visualize_outliers.py \
        --csv smoking_driking_dataset_Ver01.csv \
        --outdir assignment-1/results/outliers \
        --columns gamma_GTP triglyceride BLDS SBP DBP weight HDL_chole SGOT_ALT SGOT_AST

Notes
- IQR fences are fit on the training split ONLY to avoid data leakage.
- For specified columns, we apply log1p before computing fences and capping;
  plots are shown on the original scale (inverse-transformed) for readability.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


sns.set_context("talk")
sns.set_style("whitegrid")


@dataclass
class Fences:
    lower: float
    upper: float
    transformed: bool = False  # whether fences were computed in transformed space


def compute_iqr_fences(x: pd.Series) -> Fences:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return Fences(lower=lower, upper=upper)


def clip_with_mask(x: pd.Series, lower: float, upper: float) -> Tuple[pd.Series, pd.Series]:
    """Clip and return (clipped_series, capped_mask)."""
    capped_mask = (x < lower) | (x > upper)
    return x.clip(lower, upper), capped_mask


def maybe_log1p(x: pd.Series, do_log: bool) -> Tuple[pd.Series, Optional[str]]:
    if not do_log:
        return x, None
    # log1p is defined for x >= -1; we expect non-negative data for these features
    if (x < -1).any():
        return x, "skipped_log1p_due_to_values_below_-1"
    return np.log1p(x), None


def maybe_expm1(x: pd.Series, did_log: bool) -> pd.Series:
    return np.expm1(x) if did_log else x


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, outpath: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, dodge=False, color="#1f77b4")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize outlier handling (IQR capping, optional log1p).")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset (e.g., smoking_driking_dataset_Ver01.csv)")
    parser.add_argument(
        "--outdir",
        default="assignment-1/results/outliers",
        help="Directory to save plots and summaries.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=[
            "gamma_GTP",
            "triglyceride",
            "BLDS",
            "SBP",
            "DBP",
            "weight",
            "HDL_chole",
            "SGOT_ALT",
            "SGOT_AST",
        ],
        help="Columns to visualize (default: a curated list).",
    )
    parser.add_argument(
        "--log-columns",
        nargs="*",
        default=["gamma_GTP", "triglyceride", "BLDS", "HDL_chole", "SGOT_ALT", "SGOT_AST"],
        help="Columns to log1p-transform before fitting fences and capping.",
    )
    parser.add_argument("--train-size", type=float, default=0.8, help="Train split fraction for fitting fences.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--nonneg-columns",
        nargs="*",
        default=["sight_left", "sight_right"],
        help="Columns that should be non-negative; negatives will be set to NaN before computing fences.",
    )

    args = parser.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)

    # Split train/test indices to fit fences on train only
    idx = np.arange(len(df))
    train_idx, _ = train_test_split(idx, train_size=args.train_size, random_state=args.random_state, shuffle=True)
    train_df = df.iloc[train_idx]

    # Prepare summary
    summary_rows: List[Dict] = []

    for col in args.columns:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not in dataset. Skipping.")
            continue

        s_all = pd.to_numeric(df[col], errors="coerce")
        s_train = pd.to_numeric(train_df[col], errors="coerce")

        # Enforce non-negativity for specified columns by setting negatives to NaN
        n_neg_all = int((s_all < 0).sum()) if col in args.nonneg_columns else 0
        n_neg_train = int((s_train < 0).sum()) if col in args.nonneg_columns else 0
        if col in args.nonneg_columns:
            s_all = s_all.mask(s_all < 0, np.nan)
            s_train = s_train.mask(s_train < 0, np.nan)

        do_log = col in args.log_columns
        s_train_t, log_note = maybe_log1p(s_train, do_log)
        did_log = do_log and (log_note is None)

        fences_t = compute_iqr_fences(s_train_t.dropna())
        fences_t.transformed = did_log

        # Apply capping in the same space we computed fences
        s_all_t, log_note_all = maybe_log1p(s_all, did_log)
        s_all_t, capped_mask_t = clip_with_mask(s_all_t, fences_t.lower, fences_t.upper)

        # Compute outer (3*IQR) fences for removal accounting (train-only)
        q1_t = s_train_t.quantile(0.25)
        q3_t = s_train_t.quantile(0.75)
        iqr_t = q3_t - q1_t
        outer_lower_t = q1_t - 3.0 * iqr_t
        outer_upper_t = q3_t + 3.0 * iqr_t

        # Removal mask (values beyond outer fences in transformed space)
        s_all_t_for_remove, _ = maybe_log1p(s_all, did_log)
        remove_mask_t = (s_all_t_for_remove < outer_lower_t) | (s_all_t_for_remove > outer_upper_t)

        # For reporting in original scale
        lower_plot = maybe_expm1(pd.Series([outer_lower_t]), did_log).iloc[0]
        upper_plot = maybe_expm1(pd.Series([outer_upper_t]), did_log).iloc[0]

        # Summaries
        pct_capped = float(capped_mask_t.mean() * 100.0)
        pct_removed = float(remove_mask_t.mean() * 100.0)
        pct_transformed = 100.0 if did_log else 0.0
        summary_rows.append(
            {
                "column": col,
                "did_log1p": did_log,
                "inner_fence_lower_transformed": fences_t.lower,
                "inner_fence_upper_transformed": fences_t.upper,
                "outer_fence_lower_original_scale": lower_plot,
                "outer_fence_upper_original_scale": upper_plot,
                "%_capped": pct_capped,
                "%_removed": pct_removed,
                "%_transformed": pct_transformed,
                "negatives_set_to_nan_all": n_neg_all,
                "negatives_set_to_nan_train": n_neg_train,
                "note": (log_note or log_note_all or ""),
            }
        )

    # Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.outdir, "outlier_capping_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        # Three bar charts: transformed, capped, removed
        save_barplot(
            summary_df.sort_values("%_transformed", ascending=False),
            x="%_transformed",
            y="column",
            title="% transformed (log1p) per column",
            outpath=os.path.join(args.outdir, "percent_transformed_per_column.png"),
            xlabel="% transformed",
            ylabel="Column",
        )
        save_barplot(
            summary_df.sort_values("%_capped", ascending=False),
            x="%_capped",
            y="column",
            title="% capped (IQR inner fences) per column",
            outpath=os.path.join(args.outdir, "percent_capped_per_column.png"),
            xlabel="% capped",
            ylabel="Column",
        )
        save_barplot(
            summary_df.sort_values("%_removed", ascending=False),
            x="%_removed",
            y="column",
            title="% removed (outer fences 3·IQR) per column",
            outpath=os.path.join(args.outdir, "percent_removed_per_column.png"),
            xlabel="% removed",
            ylabel="Column",
        )


if __name__ == "__main__":
    main()
