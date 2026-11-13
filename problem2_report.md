# DNN Programming Assignment 2 — Problem 2 Report

**Transformer from scratch** (attention, encoder, HAR classification, Tiny Shakespeare LM, ablations)

All numbers below come from the executed notebook `problem2.ipynb` (seed 42) and exported files `results.csv`, `head_ablation.csv`, `context_benchmark.csv`, and `generated_samples.txt`.

---

## 1. Scaled dot-product attention

Implementation: `scores = QK^T / sqrt(d_k)`, optional boolean mask, softmax, multiply by `V`. Unit tests verify (i) attention rows sum to 1, (ii) output matches `torch.nn.functional.scaled_dot_product_attention` within `1e-5`, and (iii) an upper-triangular causal mask zeroes future attention mass.

**Hand-check (single head, \(d_k = 1\)):** With `q = k = [1, 0, -1]` and `v = [10, 20, 30]`, the score matrix is identity-like on the diagonal; softmax yields weights `[0.665, 0.245, 0.090]` for the first query (printed in the notebook). Replace with the instructor’s example if a separate spec was provided.

---

## 2. Parameter counts (multi-head attention and encoder block)

Settings: `d_model = 128`, `num_heads = 4`, `d_ff = 512`, Pre-LayerNorm block with biases enabled on all `nn.Linear` layers.

### 2.1 One multi-head self-attention module

Four projections `W_Q`, `W_K`, `W_V`, `W_O`, each `Linear(128 → 128)`:

| Component | Formula | Count |
|-----------|---------|------:|
| Weights | \(4 \times d^2\) | 65,536 |
| Biases | \(4 \times d\) | 512 |
| **MHA total** | | **66,048** |

Head count does not change parameter count when `d_model` is fixed; only `head_dim = d_model / num_heads` changes internally.

### 2.2 Full encoder block (MHA + FFN + 2× LayerNorm)

| Submodule | Count |
|-----------|------:|
| Multi-head attention | 66,048 |
| FFN: `Linear(128→512)` + `Linear(512→128)` | 131,712 |
| LayerNorm ×2 (`γ`, `β` each length 128) | 512 |
| **Block total** | **198,272** |

**Weights-only shortcut** (no biases, no LayerNorm): \(4d^2 + 2 d \cdot d_{ff} = 196{,}608\). The gap \(198{,}272 - 196{,}608 = 1{,}664\) is exactly MHA biases (512) + FFN biases (640) + LayerNorm (512).

---

## 3. Task A — HAR encoder classifier

**Setup:** UCI HAR, nine inertial channels, sequence length 128, CLS token, `d_model = 128`, 4 heads, 4 encoder layers, Adam `lr = 1e-3`, **8 epochs**, seed 42. Positional variants: `none`, `sinusoidal`, `learned`.

### 3.1 Test accuracy and training cost

| Positional encoding | Test accuracy | Macro-F1 | Parameters | Sec / epoch |
|--------------------|--------------:|---------:|-----------:|----------:|
| none | 0.871 | 0.869 | 398,982 | 0.46 |
| **sinusoidal** | **0.901** | **0.899** | 398,982 | 0.45 |
| learned | 0.888 | 0.887 | 415,494 | 0.45 |

**Best variant:** sinusoidal (highest test accuracy). Learned positional embeddings add 16,512 parameters (`128 × 128` table) but did not beat sinusoidal on this run.

**Trend:** Without positional signal, the model must infer position from channel content alone; accuracy drops ~3 points. Fixed sinusoidal features align with the 128-step window and gave the best generalization here. Learned embeddings can fit the training distribution but slightly underperformed sinusoidal on the held-out test split (8 epochs, same budget for all).

### 3.2 Comparison to Problem 1 (best: GRU)

| Model | Test accuracy | Parameters | Sec / epoch |
|-------|--------------:|-----------:|------------:|
| Transformer encoder (sinusoidal) | 0.901 | 398,982 | 0.45 |
| GRU (Problem 1) | 0.899 | 31,398 | 0.23 |

The sinusoidal Transformer **matches** the GRU test accuracy within measurement noise while using **~13× more parameters** and **~2× longer** per epoch on this hardware. The RNN remains more parameter-efficient for this sequence length; the Transformer benefits from parallel attention over the full window and explicit positional encoding.

*Figures:* positional-encoding ablation bar chart and confusion matrix for the best variant — see `problem2.ipynb` §3.

---

## 4. Task B — Tiny Shakespeare character LM

**Setup:** 90/10 split, context 128, `d_model = 128`, 4 heads, 4 layers, causal mask, AdamW `3e-4`, **10 epochs**, 300 batches/epoch, seed 42.

| Metric | Value |
|--------|------:|
| Validation loss | **2.120 nats** |
| Bits per character | **3.059** |
| Trainable parameters | 810,049 |
| Mean sec / epoch | ~2.82 |

**Sampling:** Three **500-character** continuations at temperatures **0.2**, **0.8**, and **1.5** (with `top_k = 20`) are in `generated_samples.txt`. At 0.2 text is highly repetitive; at 0.8 it mixes pseudo-dialogue and Elizabethan-like fragments; at 1.5 spelling and syntax become erratic — consistent with sharper vs flatter predictive distributions.

*Figure:* training/validation loss vs epoch — notebook §4.

---

## 5. Analysis and ablations

### 5.1 Attention heatmaps (two heads)

A 64-character validation snippet is run through the **final** encoder block with **causal** masking. Heatmaps for **head 0** and **head 1** are in notebook §5A.

**Interpretation:** Mass should appear only on and below the main diagonal (past and current tokens). Any bright upper triangle would indicate a broken mask; the plots show a strict causal pattern. Different heads assign different weights along the diagonal and to recent context — some heads emphasize local character n-grams, others spread mass over a slightly longer prefix.

### 5.2 Head-count ablation (`d_model = 128` fixed)

Short budget: 3 epochs, 150 batches/epoch for heads 1 and 8; head 4 reuses the main model’s validation loss.

| Heads | Val loss (nats) | Parameters |
|------:|----------------:|-----------:|
| 1 | 2.469 | 810,049 |
| **4** | **2.120** | 810,049 |
| 8 | 2.459 | 810,049 |

With fixed `d_model`, head count does not change total size (each head gets `head_dim = 128 / H`). **Four heads** gave the lowest loss here; a single head underfits mixing of patterns; eight heads split capacity thinly per head and did not help under this short training budget.

### 5.3 Causal mask removed

The same weights are further trained **without** a causal mask (model can attend to future tokens within the window). Evaluation:

| Setting | Val loss (nats) |
|---------|----------------:|
| Causal (proper LM) | ~2.118 |
| No-mask / “leaky” eval | ~2.438 |

The lower leaky loss is **not** a valid next-character prediction score because future characters are visible. Generation comparisons in the notebook show the no-mask model produces less coherent continuations when evaluated causally at decode time.

### 5.4 Context length vs time and GPU memory

Benchmark: batch 16, 4 layers, 4 heads, one forward pass (repeated 10×); peak allocated GPU memory recorded.

| Context length | Sec / forward | Peak GPU memory (MB) |
|---------------:|--------------:|---------------------:|
| 64 | 0.00210 | 39.9 |
| 128 | 0.00206 | 49.5 |
| 256 | 0.00205 | 80.5 |
| 512 | 0.00642 | 190.7 |

Attention memory scales **quadratically** with sequence length (\(O(L^2)\) per layer); time stays flat until 512 where cost jumps. This matches theory: longer contexts increase the attention map and activation footprint sharply (see notebook plots §5D).

---

## 6. Submission artifacts

| File | Contents |
|------|----------|
| `problem2.ipynb` | Full implementation with outputs |
| `problem2_report.md` | This report |
| `results.csv` | Required summary columns for P2 runs |
| `generated_samples.txt` | 500-char samples at T = 0.2, 0.8, 1.5 |
| `head_ablation.csv` | Head-count study |
| `context_benchmark.csv` | Context-length timing/memory |

Export figure PNGs from the notebook if the course requires separate image files in the ZIP.

---

## 7. Reproducibility

- **Seed:** 42 (Python, NumPy, PyTorch).
- **HAR:** 8 epochs per positional variant; batch size as in notebook.
- **LM:** 10 epochs × 300 batches; head ablation 3 epochs × 150 batches (comparable but shorter).
- **Environment:** See `requirements.txt`; GPU used when available (benchmark memory values are GPU-specific).
