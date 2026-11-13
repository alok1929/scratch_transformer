#!/usr/bin/env python3
"""Regenerate problem2/figures from CSV metrics (heatmaps kept from notebook if present)."""
from pathlib import Path
import shutil
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "problem2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SUB = ROOT / "2026AJ05899_DNN_Assignment" / "problem2" / "figures"


def save(name):
    p = OUT / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    if SUB.parent.exists():
        SUB.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, SUB / name)
    print("saved", p)


# --- HAR positional encoding (results.csv) ---
df = pd.read_csv(ROOT / "results.csv")
har = df[df["problem"] == "P2"].copy()
har = har[har["model"].str.contains("positional", case=False, na=False)]
if not har.empty:
    har["pe"] = har["model"].str.extract(r"positional=(\w+)")[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(har["pe"], har["test_acc"], color=["#4c72b0", "#55a868", "#c44e52"][: len(har)])
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Positional encoding")
    ax.set_title("HAR Transformer — positional encoding ablation")
    ax.set_ylim(0, 1)
    for i, row in har.iterrows():
        ax.text(row["pe"], row["test_acc"] + 0.01, f"{row['test_acc']:.3f}", ha="center", fontsize=9)
    save("har_positional_ablation.png")

# --- Head ablation ---
ha = pd.read_csv(ROOT / "head_ablation.csv")
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(ha["heads"].astype(str), ha["val_loss"], color="steelblue")
ax.set_xlabel("Number of heads")
ax.set_ylabel("Validation loss (nats)")
ax.set_title("Head-count ablation (d_model=128)")
save("head_ablation_val_loss.png")

# --- Context benchmark ---
cb = pd.read_csv(ROOT / "context_benchmark.csv")
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(cb["context_length"], cb["seconds_per_forward"], marker="o")
ax.set_xlabel("Context length")
ax.set_ylabel("Seconds per forward pass")
ax.set_title("Inference time vs context length")
save("context_time.png")

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(cb["context_length"], cb["peak_gpu_memory_mb"], marker="o", color="darkorange")
ax.set_xlabel("Context length")
ax.set_ylabel("Peak GPU memory (MB)")
ax.set_title("Peak GPU memory vs context length")
save("context_memory.png")

# log-log time scaling vs n^2 reference
fig, ax = plt.subplots(figsize=(5.5, 4))
x = cb["context_length"].values.astype(float)
y = cb["seconds_per_forward"].values
ax.loglog(x, y, "o-", label="measured")
ref = y[1] * (x / x[1]) ** 2
ax.loglog(x, ref, "--", color="gray", label=r"$\propto n^2$ (scaled)")
ax.set_xlabel("Context length")
ax.set_ylabel("Seconds per forward")
ax.set_title("Time scaling (log-log)")
ax.legend(fontsize=9)
save("context_time_loglog.png")

# --- LM summary from results ---
lm = df[(df["problem"] == "P2") & df["val_loss_nats"].notna()]
if not lm.empty:
    row = lm.iloc[0]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["val loss\n(nats)", "bits/char"], [row["val_loss_nats"], row["bits_per_char"]], color=["#8172b2", "#937860"])
    ax.set_title("Tiny Shakespeare LM (from results.csv)")
    save("lm_metrics_summary.png")

# --- Heatmaps: not in CSV; preserve from repo if available ---
for name in ("attention_heatmap_head0.png", "attention_heatmap_head1.png"):
    for src in (OUT / name, ROOT / "figures" / name, ROOT / "problem2" / "figures" / name):
        if src.resolve() != (OUT / name).resolve() and src.exists() and src.stat().st_size > 1000:
            shutil.copy2(src, OUT / name)
            if SUB.parent.exists():
                shutil.copy2(OUT / name, SUB / name)
            break
    else:
        # extract from notebook embedded outputs if needed
        nb_path = ROOT / "problem2.ipynb"
        if nb_path.exists():
            nb = json.loads(nb_path.read_text())
            heads_done = {OUT / "attention_heatmap_head0.png": False, OUT / "attention_heatmap_head1.png": False}
            hi = 0
            for cell in nb["cells"]:
                src_code = "".join(cell.get("source", []))
                if "attention_heatmap" not in src_code and "heatmap" not in src_code.lower():
                    continue
                for out in cell.get("outputs", []):
                    png = (out.get("data") or {}).get("image/png")
                    if not png:
                        continue
                    raw = base64.b64decode(png if isinstance(png, str) else "".join(png))
                    fname = f"attention_heatmap_head{hi}.png"
                    if hi < 2:
                        (OUT / fname).write_bytes(raw)
                        if SUB.parent.exists():
                            (SUB / fname).write_bytes(raw)
                        print("extracted heatmap", fname)
                        hi += 1
            break

manifest = sorted(p.name for p in OUT.glob("*.png"))
(OUT / "INDEX.md").write_text(
    "# problem2/figures (from CSV + notebook heatmaps)\n\n" + "\n".join(f"- `{n}`" for n in manifest) + "\n"
)
print("total", len(manifest), "PNGs in", OUT)
