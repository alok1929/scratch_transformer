# DNN Assignment 2 — Problem 2: Transformer from Scratch

PyTorch notebook implementing scaled dot-product attention, a custom multi-head encoder, UCI HAR classification with positional-encoding ablations, a character-level Tiny Shakespeare language model, and analysis experiments (heatmaps, head ablation, causal mask, context scaling).

Written discussion and numbers: **[problem2_report.md](problem2_report.md)**.

## Requirements

- Python 3.10+ (tested with 3.11)
- Optional: NVIDIA GPU with CUDA for faster training (CPU works but is slow)

## Quick start

```bash
cd transformer_   # this repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If `torch.cuda.is_available()` is `False` after install, pick a CUDA wheel from [pytorch.org](https://pytorch.org).

### Run the notebook

**Cursor / VS Code:** Open `problem2.ipynb`, select the `.venv` interpreter, run all cells or step through top to bottom.

**Jupyter:**

```bash
jupyter notebook problem2.ipynb
```

On first run, the notebook downloads **UCI HAR** into `UCI_HAR_data/` (gitignored) and **Tiny Shakespeare** from the web. Training defaults: HAR 8 epochs per positional variant; LM 10 epochs. Ablations use shorter budgets (see notebook §5).

Re-run the export cell at the end to refresh CSVs and `generated_samples.txt`.

## Repository layout

| Path | Description |
|------|-------------|
| `problem2.ipynb` | Implementation, tests, training, plots, exports |
| `problem2_report.md` | Report for submission |
| `requirements.txt` | Python dependencies |
| `results.csv` | Summary table (`problem`, `model`, metrics, …) |
| `generated_samples.txt` | 500-char LM samples at T = 0.2, 0.8, 1.5 |
| `head_ablation.csv` | Heads 1 / 4 / 8 validation loss |
| `context_benchmark.csv` | Forward time and peak GPU memory vs context length |
| `UCI_HAR_data/` | Downloaded dataset (not in git) |

## What the notebook covers

1. Scaled dot-product attention + unit tests (incl. PyTorch SDPA check)
2. Multi-head attention, sinusoidal PE, Pre-LayerNorm encoder block
3. **Task A:** HAR encoder — `none` / `sinusoidal` / `learned` PE; comparison to Problem 1 GRU
4. **Task B:** Causal char LM on Tiny Shakespeare (`d=128`, 4 heads, 4 layers)
5. Attention heatmaps, head-count ablation, no causal mask, context-length benchmark
6. `results.csv` and supplementary CSVs

Figures live in the executed notebook; export PNGs separately if your course ZIP requires them.

## Suggested submission bundle

- `problem2.ipynb` (with outputs saved)
- `problem2_report.md`
- `results.csv`
- `generated_samples.txt`
- Any required figure files from the notebook

## Git ignore

`.gitignore` excludes `UCI_HAR_data/`, `uci_har_data/`, and `.venv/`.
