#!/usr/bin/env python3
"""Merge Problem 1 and Problem 2 result tables into submission results.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ASSIGNMENT_COLS = [
    "problem",
    "model",
    "seed",
    "seq_len",
    "test_acc",
    "macro_f1",
    "test_mse",
    "params",
    "sec_per_epoch",
    "val_loss_nats",
    "bits_per_char",
    "notes",
]


def load_problem1(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    for _, r in raw.iterrows():
        kind = str(r.get("kind", "")).lower()
        task = str(r.get("task", "")).lower()
        exp = str(r.get("exp", ""))
        if task == "har":
            model = kind.upper()
            notes = f"HAR; h={int(r['H'])}; exp={exp}"
            seq_len = int(r["H"]) if not pd.isna(r.get("H")) else np.nan
            macro_f1 = r.get("test_f1", np.nan)
        elif task == "adding":
            model = kind.upper()
            seq_len = int(r["T"]) if not pd.isna(r.get("T")) else np.nan
            macro_f1 = np.nan
            notes = f"adding T={seq_len}; exp={exp}"
        else:
            model = kind.upper()
            seq_len = np.nan
            macro_f1 = np.nan
            notes = exp
        if "forget" in exp or not pd.isna(r.get("forget_bias")):
            fb = r.get("forget_bias")
            if not pd.isna(fb):
                notes = (notes or "") + f"; forget_bias={fb}"
        rows.append(
            {
                "problem": 1,
                "model": model,
                "seed": int(r["seed"]) if not pd.isna(r.get("seed")) else np.nan,
                "seq_len": seq_len,
                "test_acc": r.get("test_acc", np.nan),
                "macro_f1": macro_f1,
                "test_mse": r.get("test_mse", np.nan),
                "params": r.get("params", np.nan),
                "sec_per_epoch": r.get("sec_per_epoch", np.nan),
                "val_loss_nats": np.nan,
                "bits_per_char": np.nan,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=ASSIGNMENT_COLS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--p1",
        type=Path,
        default=Path("problem1/problem1_outputs/results.csv"),
    )
    p.add_argument("--p2", type=Path, default=Path("results_p2.csv"))
    p.add_argument("-o", "--output", type=Path, default=Path("results.csv"))
    args = p.parse_args()

    frames = [load_problem1(args.p1)]
    if args.p2.exists():
        p2 = pd.read_csv(args.p2)
        if "problem" in p2.columns:
            p2["problem"] = p2["problem"].replace({"P1": 1, "P2": 2, "1": 1, "2": 2})
        frames.append(p2.reindex(columns=ASSIGNMENT_COLS))
    else:
        print(f"Warning: {args.p2} not found; output will contain Problem 1 rows only.")

    out = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
