# DNN Programming Assignment Report

**Name:** *(fill)* · **Student ID:** 2026AJ05899 · **Date:** 13 November 2026

---

## Problem 1 — RNN, LSTM, GRU on sequences

### Setup

**UCI HAR:** 128 × 9 inertial channels, 6 classes; train subjects 21, test 9; validation = 4 held-out train subjects (`rng.choice(..., 4)` per seed). Per-channel z-score using **training-only** statistics. **Adding problem:** `make_adding` with two markers per half; train/val/test 10k/1k/1k at seeds 0/1/2 for data, **training seeds 7, 21, 42**. Hidden size **h = 96** (variant table). Adam lr **1e-3**, gradient clip **1.0**, HAR **20** epochs (batch 64), adding **100** epochs (batch 128).

### Cell verification and parameter counts

`MyRNNCell`, `MyLSTMCell`, `MyGRUCell` implemented with matrix ops; weights copied into matching `nn.RNN` / `nn.LSTM` / `nn.GRU` and outputs matched within **1e-5** on random batches (see `problem1.ipynb` §Task 2).

For **h = 96**, **d = 9**: hand counts RNN **10 176**, LSTM **40 704**, GRU **30 528** (single bias per gate). PyTorch adds separate **b_ih** and **b_hh** (+96 / +384 / +288), giving **10 272 / 41 088 / 30 816** (Table 1).

**Table 1 — Parameter reconciliation (h = 96, d = 9).**

| Model | By hand | Code (1 bias) | PyTorch (2 biases) |
|-------|--------:|--------------:|-------------------:|
| RNN | 10 176 | 10 272 | 10 272 |
| LSTM | 40 704 | 41 088 | 41 088 |
| GRU | 30 528 | 30 816 | 30 816 |

### HAR classification

Three models, **3 seeds** each, same budget. **Table 2** (mean ± sample std over seeds).

**Table 2 — HAR test accuracy and cost (h = 96).**

| Model | Test acc | Macro-F1 | Params | s/epoch |
|-------|----------|----------|-------:|--------:|
| RNN | 0.8471 ± 0.0212 | 0.8460 ± 0.0216 | 10 854 | 0.27 |
| LSTM | 0.8911 ± 0.0076 | 0.8913 ± 0.0079 | 41 670 | 0.35 |
| GRU | **0.8991 ± 0.0054** | **0.8991 ± 0.0053** | 31 398 | **0.23** |

**Figure 1** (`problem1/figures/har_class_balance.png`) — class balance. **Figure 2** (`har_confusion.png`) — confusion matrix for best GRU run. Gated models clearly beat vanilla RNN; GRU is best accuracy with fewer parameters than LSTM and fastest per epoch.

### Adding problem

**Table 3 — Test MSE (mean ± std) and solved runs (MSE &lt; 0.05).**

| T | RNN | LSTM | GRU |
|---|-----|------|-----|
| 50 | 0.0071 (3/3) | 0.0001 (3/3) | 0.0001 (3/3) |
| 100 | 0.1675 (0/3) | 0.0004 (3/3) | 0.0001 (3/3) |
| 200 | 0.1682 (0/3) | 0.0008 (3/3) | 0.0001 (3/3) |
| 400 | 0.1536 (0/3) | 0.0653 (2/3) | 0.0002 (3/3) |

Baseline constant 1.0 ≈ **0.167 MSE**. **Figure 3** (`adding_mse_and_curves.png`) — MSE vs T and learning curves at T = 200. Plain RNN fails beyond short T (scores stay near baseline); LSTM/GRU retain long-range sums.

### Gradients, sweep, ablation

**Figure 4** (`gradient_norms.png`) — ‖∂L/∂h_t‖ vs t (log scale) at T = 100 before/after training: RNN norms vanish early; gated cells maintain larger backward signals.

**Figure 5** (`hidden_size_sweep.png`) — LSTM/GRU HAR accuracy vs hidden size {16,32,64,128}; gains taper after 64–96.

**Figure 6** (`ablation_forget_bias.png`) — **Odd-digit ablation:** LSTM forget bias 0 vs 1 on adding at **T = 200**; bias 1 improves stability and lower MSE (3 seeds).

### P1 analysis (wearable)

For a **battery-powered wearable**, **GRU** is the pragmatic choice: **highest HAR accuracy** (Table 2), **smallest footprint among gated models** (31k params vs 42k LSTM), and **fastest epochs** (0.23 s). Transformer-style models are unnecessary on-device for this 128-step sensor window unless cloud offload is assumed.

---

## Problem 2 — Transformer from scratch

### Attention and encoder block

Scaled dot-product attention implemented with mask support; tests pass (row sums 1, matches `scaled_dot_product_attention` within **1e-5**, causal mask zeros future weights). **d = 128**, **4 heads**, **d_ff = 512**: MHA **66 048** params; full Pre-LN block **198 272** (reconciles 4d² + 2·d·d_ff plus biases and LayerNorm).

### Task A — HAR encoder

CLS token, 4 encoder layers; positional variants **none / sinusoidal / learned**, 8 epochs, seed 42.

**Table 4 — Positional-encoding ablation.**

| PE | Test acc | Macro-F1 | Params | s/epoch |
|----|----------|----------|-------:|--------:|
| none | 0.871 | 0.869 | 398 982 | 0.46 |
| **sinusoidal** | **0.901** | **0.899** | 398 982 | 0.45 |
| learned | 0.888 | 0.887 | 415 494 | 0.45 |

**Figure 7** (`problem2/figures/fig_p2_02.png`) — validation accuracy by PE. Sinusoidal matches **GRU 0.899** (Table 2) with ~13× parameters and ~2× epoch time — attention wins on accuracy tie but RNN wins on efficiency.

### Task B — Tiny Shakespeare LM

Context 128, 4 layers, causal mask; val loss **2.120 nats** (**3.059** bits/char). **Figure 8** (`fig_p2_04.png`) — train/val loss. Samples in `generated_samples.txt`: T = 0.2 repetitive; 0.8 more diverse; 1.5 noisy — as expected.

### Ablations

**Figures 9–10** (`fig_p2_05.png`, `fig_p2_06.png`) — final-block attention heads 0–1; mass respects causality (lower triangle). **Heads 1/4/8:** val loss **2.469 / 2.120 / 2.459** nats (4 best). **No causal mask:** leaky eval loss **2.438** is not a valid LM metric; generation degrades. **Context 64–512:** time modest until 512; GPU memory **~40 → 191 MB** — consistent with **O(L²)** attention (see `context_benchmark.csv`, **Figure 11** `fig_p2_07.png`–`fig_p2_08.png`).

---

## AI-use statement

AI assistance (Cursor / Claude) was used for: structuring notebooks, debugging training loops, drafting this report and README, and automating submission file layout. All experiments were run locally; numbers come from `results.csv` and saved notebook outputs. I understand and can explain every submitted file.

## References

- UCI HAR dataset (ID 240); Karpathy Tiny Shakespeare `input.txt`.
- Vaswani et al., *Attention Is All You Need* (positional encoding and Transformer block).

---

*Export to PDF: `pandoc report.md -o report.pdf` (A4, 11pt) and keep within 6 pages; trim if needed.*
