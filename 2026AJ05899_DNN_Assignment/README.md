# DNN Programming Assignment — RNN, LSTM, GRU and Transformer

**Student ID:** 2026AJ05899 · **Hidden size (P1):** 96 · **Training seeds:** 7, 21, 42  
**Individualisation (last digit 9, odd):** LSTM forget-gate bias **0 vs 1** at adding length T = 200.

## Environment

- Python **3.11**, PyTorch **2.x**, NumPy, pandas, Matplotlib, scikit-learn (see `requirements.txt`).
- **GPU:** T4 (or similar) recommended for Problem 2 LM; Problem 1 runs in ~15–25 min on T4, Problem 2 ~25–35 min on T4 (8 HAR epochs + 10 LM epochs + short ablations).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
```

## How to run

| Notebook | Command | Notes |
|----------|---------|--------|
| `problem1/problem1.ipynb` | Open → **Restart & Run All** | Downloads UCI HAR; writes `results.csv` in cwd; saves figures under `figures/`. Skips finished runs if `results.csv` already exists. |
| `problem2/problem2.ipynb` | Open → **Restart & Run All** | Downloads HAR + Tiny Shakespeare; writes `generated_samples.txt`, `results.csv` (if run from parent), plots in notebook and `problem2/figures/` after export. |

Use **relative paths only** (no Drive paths). Seed **42** for Problem 2 main runs; Problem 1 uses seeds **7, 21, 42**.

**Epoch budgets (stated in report):** P1 HAR **20** epochs, batch 64, Adam lr **1e-3**, grad clip **1.0**; adding problem **100** epochs, batch 128. P2 HAR **8** epochs; Shakespeare LM **10** epochs × 300 batches; head ablation **3** epochs.

## Folder layout (this zip)

```
README.md
requirements.txt
report.pdf          ← submit this PDF (source: report.md)
results.csv         ← P1 + P2 rows, assignment columns
problem1/
  problem1.ipynb
  figures/          ← class balance, confusion, adding curves, gradients, sweep, ablation
problem2/
  problem2.ipynb
  generated_samples.txt
  figures/          ← plots exported from executed notebook
```

Do **not** ship `UCI_HAR_data/`, `.venv/`, or checkpoints.

## Results summary

- **P1 best HAR:** GRU test acc **0.899** (mean ± std over seeds), **31 398** params, **0.23 s/epoch**.
- **P2 best HAR encoder:** sinusoidal PE test acc **0.901**, **398 982** params (see `results.csv` and `report.pdf`).
- **P2 LM:** val loss **2.120 nats**, **3.059** bits/char (seed 42).

## Known issues

- If HAR download fails offline, place the UCI zip manually as described in the assignment PDF.
- Problem 1 notebook should be saved **with outputs** after Run All before zipping (portal requirement).
