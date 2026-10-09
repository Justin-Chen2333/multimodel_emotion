"""Day 18: figures for the README.

Draws
  docs/architecture.png          model pipeline (Whisper -> RoBERTa || wav2vec2 -> late fusion)
  results/day18_layer_weights.png  learned wav2vec2 layer weights vs the Day 9 linear probe

Reads only small files that are committed to the repo (results/*_summary.json,
results/day9_layer_probe.csv, results/fusion_best_temperature.json), so it runs
without data, GPU or checkpoints:  python src/figures_day18.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ROOT / "docs"

TEXT_C = "#2f6fb0"   # text branch
AUDIO_C = "#c2702a"  # audio branch
FUSE_C = "#4a4a4a"   # fusion head
INK = "#222222"


# ----------------------------------------------------------------------------- architecture
def box(ax, x, y, w, h, title, sub="", color=INK, fill="white", dashed=False):
    """Rounded box centred at (x, y); title in bold, optional smaller sub-line."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=color, facecolor=fill,
        linestyle="--" if dashed else "-"))
    if sub:
        two = "\n" in sub                 # two-line sub-text: push the title up
        ax.text(x, y + (0.27 if two else 0.13), title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=color)
        ax.text(x, y - (0.12 if two else 0.17), sub, ha="center", va="center", fontsize=8.5, color=INK)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=color)


def arrow(ax, p, q, color=INK, label="", rad=0.0, label_xy=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        lx, ly = label_xy if label_xy else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.16)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def draw_architecture(path, temperature):
    fig, ax = plt.subplots(figsize=(15.5, 4.9))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 4.9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    yt, ya, ym = 3.75, 1.15, 2.45      # text row, audio row, fusion row

    # input
    box(ax, 0.85, ym, 1.4, 0.9, "Speech", "16 kHz mono")

    # text branch
    box(ax, 3.0, yt, 2.1, 0.85, "Whisper-small", "ASR (demo only)", TEXT_C, dashed=True)
    box(ax, 5.85, yt, 2.7, 0.85, "RoBERTa-base", "fine-tuned, lr 2e-5", TEXT_C)
    box(ax, 8.55, yt, 1.9, 0.85, "LayerNorm", "<s> vector → 768", TEXT_C)

    # audio branch
    box(ax, 3.0, ya, 2.1, 0.85, "wav2vec2-base", "frozen, pre-extracted", AUDIO_C)
    box(ax, 5.85, ya, 2.7, 0.85, "13 hidden states", "mean-pool over time → 13×768", AUDIO_C)
    box(ax, 8.55, ya, 1.9, 0.85, "Σ softmax(w)·h", "layer mix + LN → 768", AUDIO_C)

    # fusion head
    box(ax, 10.55, ym, 1.1, 0.9, "concat", "1536", FUSE_C)
    box(ax, 12.45, ym, 2.15, 1.15, "MLP head",
        "Dropout → 1536→256\nReLU → Dropout → 256→7", FUSE_C)
    box(ax, 14.55, ym, 1.6, 0.9, "7 emotions", f"softmax(z / {temperature:.2f})", FUSE_C)

    # arrows: input -> branches
    arrow(ax, (1.58, ym + 0.25), (1.93, yt - 0.1), TEXT_C, rad=-0.2)
    arrow(ax, (1.58, ym - 0.25), (1.93, ya + 0.1), AUDIO_C, rad=0.2)
    arrow(ax, (4.08, yt), (4.48, yt), TEXT_C)
    arrow(ax, (7.23, yt), (7.58, yt), TEXT_C)
    arrow(ax, (4.08, ya), (4.48, ya), AUDIO_C)
    arrow(ax, (7.23, ya), (7.58, ya), AUDIO_C)
    arrow(ax, (9.52, yt - 0.2), (10.08, ym + 0.35), TEXT_C, rad=-0.15)
    arrow(ax, (9.52, ya + 0.2), (10.08, ym - 0.35), AUDIO_C, rad=0.15)
    arrow(ax, (11.12, ym), (11.35, ym), FUSE_C)
    arrow(ax, (13.55, ym), (13.73, ym), FUSE_C)

    # note on where the transcript comes from
    ax.text(4.45, yt + 0.72,
            "training / evaluation: gold MELD transcript    ·    demo: Whisper transcript",
            ha="center", va="center", fontsize=8.5, color=TEXT_C, style="italic")
    ax.text(7.75, 0.25,
            "Late fusion: each modality is encoded separately and combined only at the classifier. "
            "Single-modality ablations use the same head with one branch (input 768).",
            ha="center", va="center", fontsize=8.5, color="#555555")

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- layer weights
def load_weights(tag):
    p = RES / f"{tag}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("layer_weights_mean")


def load_probe():
    p = RES / "day9_layer_probe.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r["layer"].isdigit()]
    return [float(r["dev_wF1"]) for r in sorted(rows, key=lambda r: int(r["layer"]))]


def draw_layer_weights(path):
    series = [
        ("audio-only (20 epochs)", "day13_audio_e20", AUDIO_C, "o"),
        ("text + audio, RoBERTa frozen", "day12_frozen", "#7a5195", "s"),
        ("text + audio, RoBERTa fine-tuned", "day12", TEXT_C, "^"),
    ]
    probe = load_probe()
    layers = list(range(13))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    for name, tag, color, marker in series:
        w = load_weights(tag)
        if w is None:
            print(f"  skip {tag}: results/{tag}_summary.json not found")
            continue
        ax.plot(layers, w, marker=marker, color=color, linewidth=1.8, markersize=5, label=name)
    ax.axhline(1 / 13, color="#999999", linestyle="--", linewidth=1, label="uniform (1/13)")
    ax.set_xticks(layers)
    ax.set_xlabel("wav2vec2-base layer (0 = CNN output)")
    ax.set_ylabel("learned softmax weight")
    ax.set_title("Learned layer weights (mean of 3 seeds)", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    if probe is not None:
        ax.bar(layers, probe, color=AUDIO_C, alpha=0.8)
        ax.set_ylim(min(probe) - 0.03, max(probe) + 0.02)
    ax.set_xticks(layers)
    ax.set_xlabel("wav2vec2-base layer")
    ax.set_ylabel("dev weighted F1")
    ax.set_title("Day 9 linear probe per layer (1 run)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    DOCS.mkdir(exist_ok=True)
    t_path = RES / "fusion_best_temperature.json"
    temperature = json.loads(t_path.read_text(encoding="utf-8"))["temperature"] if t_path.exists() else 1.857
    draw_architecture(DOCS / "architecture.png", temperature)
    print("wrote", (DOCS / "architecture.png").relative_to(ROOT))
    draw_layer_weights(RES / "day18_layer_weights.png")
    print("wrote", (RES / "day18_layer_weights.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
