# DNN Assignment (Problems 1 & 2)

Course work: **RNN / LSTM / GRU** (Problem 1) and **Transformer from scratch** (Problem 2).

## Submission package (upload this)

Everything for the portal zip lives in:

**[`2026AJ05899_DNN_Assignment/`](2026AJ05899_DNN_Assignment/)**

Zip that folder as **`2026AJ05899_DNN_Assignment.zip`** (≤ 25 MB, no datasets or `.venv`).

| Path | Role |
|------|------|
| `README.md` | How to run both notebooks, variant, runtime |
| `requirements.txt` | Pinned dependencies |
| `report.md` / **`report.pdf`** | Written report (PDF required on portal) |
| `results.csv` | Merged P1 + P2 runs |
| `problem1/problem1.ipynb` + `figures/` | Problem 1 |
| `problem2/problem2.ipynb` + `generated_samples.txt` + `figures/` | Problem 2 |

**Before zipping:** Run **Restart & Run All** on both notebooks and save **with outputs**. Convert `report.md` → `report.pdf` if `report.pdf` is missing:

```bash
pandoc report.md -o report.pdf -V geometry:margin=2cm -V fontsize=11pt
```

## Workspace (development)

| Path | Notes |
|------|--------|
| `problem1/` | P1 notebook + `problem1_outputs/` (training artifacts) |
| `problem2.ipynb` | Main P2 notebook (copy also in submission folder) |
| `problem2_report.md` | Extended P2-only notes (superset of report §P2) |
| `pronlem2.2/` | Old duplicate — **do not submit** |

Assignment spec: `DNN Programming Assignment – RNN, LSTM, GRU and Transformer (1).docx`
