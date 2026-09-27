"""
Day 6 — 文本基线：训练 5 epochs + dev 评估 + 保存最佳权重 + 混淆矩阵 / 错误分析 + class weights 对比

先装（只需一次）：
    python -m pip install scikit-learn

用法（Anaconda Prompt，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python -m text_day6 --limit 320            # 冒烟测试，1 分钟内跑完（不存权重；输出写到 day6_smoke_*，不覆盖正式结果）
    python -m text_day6                        # 今日最低目标：不加权、seed 42、5 epochs（约 2–3 分钟）
    python -m text_day6 --sweep                # 对比：{none, balanced, sqrt} × seed {42,43,44} = 9 次（约 20–25 分钟）
    python -m text_day6 --weights sqrt --seeds 42 43   # 自选组合
    python -m text_day6 --lr 1e-5              # 调学习率（计划建议 1e-5 ~ 2e-5）

设定沿用 Day 5：AdamW lr 2e-5 / wd 0.01 / 10% warmup + 线性衰减 / 梯度裁剪 1.0 / bf16 / batch 16 / max_length 128。
class weights（只影响训练 loss；dev loss 一律不加权，才能横向比较）：
    none      不加权
    balanced  N / (K * n_c)，最大/最小约 17 倍
    sqrt      sqrt(balanced)，约 4 倍，折中
选模型：每次运行按 dev weighted F1 选最佳 epoch；本次所有运行里 dev weighted F1 最高的那一个存成
    results\\text_best.pt（纯 state_dict，TextClassifier().load_state_dict 直接加载）+ results\\text_best.json（配置和分数）
    如果已有 text_best.json 且分数更高，不覆盖（--force-save 强制覆盖）
输出：
    results\\day6_history.csv       每次运行每个 epoch 的 train/dev 指标
    results\\day6_runs.csv          每次运行的最佳 epoch 指标（含各类 F1）
    results\\day6_curves.png        dev weighted F1 曲线 + 最佳运行的 loss 曲线
    results\\day6_dev_report.txt    最佳运行在 dev 上的 classification_report + 混淆矩阵
    results\\day6_confusion.png     混淆矩阵（按行归一化 = 每个真实类别被预测成什么）
    results\\day6_dev_errors.csv    dev 上所有错例，按置信度从高到低
    results\\day6_dev_probs.npy     最佳运行在 dev 上的概率 (1109, 7)，Day 13 和多模态逐类对比用
"""
import argparse
import json
import math
import os
import random
import time
from functools import partial

# 必须在 import transformers 之前设置
os.environ.setdefault("HF_HOME", r"F:\hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
except ImportError:
    raise SystemExit("缺 scikit-learn：python -m pip install scikit-learn")

from roberta_day4 import LABELS, MODEL_NAME, OUT_DIR, TextClassifier, load_split

try:  # 本机 Day 5 文件叫 epoch1_day5.py，项目里叫 text_day5.py
    from epoch1_day5 import TextDataset, collate, f1_report
except ImportError:
    from text_day5 import TextDataset, collate, f1_report

K = len(LABELS)
WEIGHT_KINDS = ["none", "balanced", "sqrt"]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def class_weights(kind, labels):
    if kind == "none":
        return None
    counts = np.bincount(labels, minlength=K)
    w = len(labels) / (K * counts)
    if kind == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w, dtype=torch.float)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs, golds = [], []
    for ids, mask, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(ids.to(device), mask.to(device))
        probs.append(logits.float().softmax(-1).cpu())
        golds.append(y)
    return torch.cat(probs).numpy(), torch.cat(golds).numpy()


def metrics(probs, gold):
    pred = probs.argmax(1)
    labels = list(range(K))
    return dict(
        loss=float(-np.log(probs[np.arange(len(gold)), gold] + 1e-12).mean()),
        acc=float((pred == gold).mean()),
        wf1=float(f1_score(gold, pred, labels=labels, average="weighted", zero_division=0)),
        mf1=float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        per_class=f1_score(gold, pred, labels=labels, average=None, zero_division=0).tolist(),
        pred=pred,
    )


def train_one(kind, seed, a, train_ds, train_labels, dev_dl, device, amp, pad_id):
    set_seed(seed)
    model = TextClassifier().to(device)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                          collate_fn=partial(collate, pad_id=pad_id), generator=g)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": a.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    total = len(train_dl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * a.warmup), total)

    w = class_weights(kind, train_labels)
    if w is not None:
        print("  class weights:", {l: round(float(x), 2) for l, x in zip(LABELS, w)})
        w = w.to(device)

    hist, best, best_state = [], None, None
    print(f"  {'epoch':>5} {'train_loss':>10} {'dev_loss':>8} {'dev_acc':>7} {'dev_wF1':>7} {'dev_mF1':>7}  秒")
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(ids, mask)
            loss = F.cross_entropy(logits.float(), y, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()

        probs, gold = predict(model, dev_dl, device, amp)
        m = metrics(probs, gold)
        secs = time.time() - t0
        improved = best is None or m["wf1"] > best["wf1"]
        print(f"  {ep:>5} {run / len(train_dl):>10.4f} {m['loss']:>8.4f} {m['acc']:>7.4f} "
              f"{m['wf1']:>7.4f} {m['mf1']:>7.4f}  {secs:.0f}{'  ★' if improved else ''}")
        hist.append(dict(weights=kind, seed=seed, epoch=ep, train_loss=round(run / len(train_dl), 5),
                         dev_loss=round(m["loss"], 5), dev_acc=round(m["acc"], 5),
                         dev_wf1=round(m["wf1"], 5), dev_mf1=round(m["mf1"], 5), secs=round(secs, 1)))
        if improved:
            best = dict(epoch=ep, probs=probs, gold=gold, **m)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return hist, best, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--weights", nargs="+", default=["none"], choices=WEIGHT_KINDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--sweep", action="store_true", help="= --weights none balanced sqrt --seeds 42 43 44")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟测试，不存权重）")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--force-save", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        a.weights = WEIGHT_KINDS
        if a.seeds == [42]:
            a.seeds = [42, 43, 44]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.no_amp
    print(f"device={device}  bf16={amp}  HF_HOME={os.environ['HF_HOME']}")
    OUT_DIR.mkdir(exist_ok=True)
    lm = OUT_DIR / "label_map.json"
    if lm.exists():
        saved = json.loads(lm.read_text(encoding="utf-8"))["labels"]
        assert saved == LABELS, f"label 顺序和 label_map.json 不一致：{saved} vs {LABELS}"

    # ---------- 数据 ----------
    train_df, dev_df = load_split("train"), load_split("dev")
    if a.limit:
        train_df = train_df.sample(n=min(a.limit, len(train_df)), random_state=42)
        dev_df = dev_df.head(a.limit)
    dev_df = dev_df.reset_index(drop=True)
    tok = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(train_df, tok, a.max_length)
    dev_ds = TextDataset(dev_df, tok, a.max_length)
    dev_dl = DataLoader(dev_ds, batch_size=64, shuffle=False,
                        collate_fn=partial(collate, pad_id=tok.pad_token_id))
    train_labels = train_df["label"].to_numpy()
    print(f"train {len(train_ds)} / dev {len(dev_ds)}")

    # sklearn 与 Day 5 手写 F1 核对（用一组随机预测）
    rng = np.random.default_rng(0)
    gold_chk = dev_df["label"].to_numpy()
    pred_chk = rng.integers(0, K, len(gold_chk))
    mine, _, _ = f1_report(pred_chk, gold_chk)
    skl = f1_score(gold_chk, pred_chk, labels=list(range(K)), average="weighted", zero_division=0)
    print(f"weighted F1 核对：Day 5 手写 {mine:.6f} vs sklearn {skl:.6f}  "
          f"{'一致 ✅' if abs(mine - skl) < 1e-9 else '不一致 ⚠️'}")

    n_runs = len(a.weights) * len(a.seeds)
    tag = "day6_smoke" if a.limit else "day6"  # 冒烟测试的输出另存，不覆盖正式结果
    print(f"\n共 {n_runs} 次运行：weights={a.weights} × seeds={a.seeds}，每次 {a.epochs} epochs")

    # ---------- 训练 ----------
    all_hist, runs = [], []
    best_run, best_state = None, None
    t_all = time.time()
    for kind in a.weights:
        for seed in a.seeds:
            print(f"\n========== weights={kind}  seed={seed}  ({len(runs) + 1}/{n_runs}) ==========")
            hist, best, state = train_one(kind, seed, a, train_ds, train_labels, dev_dl, device, amp,
                                          tok.pad_token_id)
            all_hist += hist
            runs.append(dict(weights=kind, seed=seed, best_epoch=best["epoch"],
                             dev_acc=round(best["acc"], 4), dev_wf1=round(best["wf1"], 4),
                             dev_mf1=round(best["mf1"], 4),
                             **{f"f1_{l}": round(best["per_class"][i], 4) for i, l in enumerate(LABELS)}))
            print(f"  → 最佳 epoch {best['epoch']}：acc {best['acc']:.4f}  weighted F1 {best['wf1']:.4f}  "
                  f"macro F1 {best['mf1']:.4f}")
            if best_run is None or best["wf1"] > best_run["wf1"]:
                best_run = dict(weights=kind, seed=seed, **best)
                best_state = state
            else:
                del state
    print(f"\n全部训练用时 {(time.time() - t_all) / 60:.1f} 分钟")
    if device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    pd.DataFrame(all_hist).to_csv(OUT_DIR / f"{tag}_history.csv", index=False)
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(OUT_DIR / f"{tag}_runs.csv", index=False)

    # ---------- 汇总 ----------
    print("\n========== 汇总（每次运行取最佳 epoch；多个 seed 给均值 ± 标准差）==========")

    def ms(s):
        return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}" if len(s) > 1 else f"{s.mean():.4f}"

    print(f"{'weights':9s} {'n':>2} {'dev acc':>17} {'dev weighted F1':>17} {'dev macro F1':>17}"
          f" {'neutral':>7} {'disgust':>7} {'fear':>7}  best epochs")
    for kind, g in runs_df.groupby("weights", sort=False):
        print(f"{kind:9s} {len(g):>2} {ms(g.dev_acc):>17} {ms(g.dev_wf1):>17} {ms(g.dev_mf1):>17}"
              f" {g.f1_neutral.mean():>7.3f} {g.f1_disgust.mean():>7.3f} {g.f1_fear.mean():>7.3f}"
              f"  {g.best_epoch.tolist()}")
    print("（记下这里的数字填消融表：计划 Day 6 checklist 要的 val accuracy 就是 dev acc）")

    # ---------- 保存最佳权重 ----------
    meta = dict(weights=best_run["weights"], seed=best_run["seed"], epoch=best_run["epoch"],
                dev_acc=round(best_run["acc"], 4), dev_wf1=round(best_run["wf1"], 4),
                dev_mf1=round(best_run["mf1"], 4),
                per_class_f1={l: round(best_run["per_class"][i], 4) for i, l in enumerate(LABELS)},
                lr=a.lr, batch=a.batch, epochs=a.epochs, max_length=a.max_length, model=MODEL_NAME,
                labels=LABELS, saved_at=time.strftime("%Y-%m-%d %H:%M"))
    print(f"\n本次最佳运行：weights={meta['weights']} seed={meta['seed']} epoch {meta['epoch']}  "
          f"dev weighted F1 {meta['dev_wf1']}")
    pt, js = OUT_DIR / "text_best.pt", OUT_DIR / "text_best.json"
    if a.limit or a.no_save:
        print("（--limit / --no-save：不保存权重）")
    else:
        old = json.loads(js.read_text(encoding="utf-8")) if js.exists() else None
        if old and old["dev_wf1"] >= meta["dev_wf1"] and not a.force_save:
            print(f"已有 text_best.pt 的 dev weighted F1 {old['dev_wf1']}（{old['weights']}/seed {old['seed']}）"
                  f" ≥ 本次，保留旧的（--force-save 可覆盖）")
        else:
            torch.save(best_state, pt)
            js.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"已保存 {pt}（{pt.stat().st_size / 1e6:.0f} MB）和 {js.name}")

    # ---------- 最佳运行的 dev 分析 ----------
    probs, gold, pred = best_run["probs"], best_run["gold"], best_run["pred"]
    np.save(OUT_DIR / f"{tag}_dev_probs.npy", probs)
    report = classification_report(gold, pred, labels=list(range(K)), target_names=LABELS,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=list(range(K)))
    short = [l[:4] for l in LABELS]
    cm_txt = "gold\\pred " + " ".join(f"{s:>5}" for s in short) + "\n" + "\n".join(
        f"{LABELS[i]:9s} " + " ".join(f"{v:>5d}" for v in row) for i, row in enumerate(cm))
    print(f"\n========== classification report（dev，{meta['weights']}/seed {meta['seed']}）==========")
    print(report)
    print("混淆矩阵（行 = 真实，列 = 预测）：")
    print(cm_txt)
    (OUT_DIR / f"{tag}_dev_report.txt").write_text(
        f"{json.dumps(meta, ensure_ascii=False)}\n\n{report}\n{cm_txt}\n", encoding="utf-8")

    print("\n最常见的混淆（真实 → 预测）：")
    pairs = [(cm[i, j], i, j) for i in range(K) for j in range(K) if i != j and cm[i, j] > 0]
    for n, i, j in sorted(pairs, reverse=True)[:8]:
        print(f"  {LABELS[i]:9s} → {LABELS[j]:9s} {n:4d}  （占该真实类 {n / cm[i].sum():.0%}）")

    words = dev_df["text"].str.split().str.len()
    bucket = pd.cut(words, [0, 2, 5, 10, 1000], labels=["1–2 词", "3–5 词", "6–10 词", "11+ 词"])
    print("\n按句长的 dev 准确率：")
    for b, idx in dev_df.groupby(bucket, observed=True).groups.items():
        idx = np.asarray(idx)
        print(f"  {b:7s} {len(idx):4d} 条  acc {(pred[idx] == gold[idx]).mean():.3f}")

    err = dev_df.assign(gold=[LABELS[i] for i in gold], pred=[LABELS[i] for i in pred],
                        conf=probs.max(1).round(3), gold_prob=probs[np.arange(len(gold)), gold].round(3))
    err = err[err.gold != err.pred].sort_values("conf", ascending=False)
    err[["Dialogue_ID", "Utterance_ID", "Speaker", "text", "gold", "pred", "conf", "gold_prob"]].to_csv(
        OUT_DIR / f"{tag}_dev_errors.csv", index=False, encoding="utf-8-sig")
    print(f"\n错例 {len(err)} 条已存 {tag}_dev_errors.csv；置信度最高的 8 条：")
    for r in err.head(8).itertuples():
        print(f"  [{r.gold} → {r.pred} {r.conf:.2f}] dia{r.Dialogue_ID}_utt{r.Utterance_ID}  {r.text}")

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = pd.DataFrame(all_hist)
        colors = dict(none="tab:blue", balanced="tab:red", sqrt="tab:green")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for (kind, seed), g in h.groupby(["weights", "seed"], sort=False):
            ax[0].plot(g.epoch, g.dev_wf1, marker="o", color=colors[kind], alpha=0.8, label=f"{kind}/s{seed}")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("dev weighted F1"); ax[0].set_title("dev weighted F1")
        ax[0].legend(fontsize=7, ncol=2)
        g = h[(h.weights == best_run["weights"]) & (h.seed == best_run["seed"])]
        ax[1].plot(g.epoch, g.train_loss, marker="o", label="train loss (with class weights, if any)")
        ax[1].plot(g.epoch, g.dev_loss, marker="o", label="dev loss (unweighted)")
        ax[1].axhline(math.log(K), ls="--", c="gray", lw=0.8, label="ln 7")
        ax[1].set_xlabel("epoch"); ax[1].set_title(f"loss — best run ({best_run['weights']}/s{best_run['seed']})")
        ax[1].legend(fontsize=8)
        for x in ax:
            x.set_xticks(range(1, a.epochs + 1))
        plt.tight_layout(); plt.savefig(OUT_DIR / f"{tag}_curves.png", dpi=120); plt.close()

        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(K), LABELS, rotation=45, ha="right"); ax.set_yticks(range(K), LABELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("gold")
        ax.set_title(f"Text-only RoBERTa, dev (row-normalised)\nweighted F1 {meta['dev_wf1']}")
        fig.colorbar(im, fraction=0.046); plt.tight_layout()
        plt.savefig(OUT_DIR / f"{tag}_confusion.png", dpi=130); plt.close()
        print(f"\n已保存 {tag}_curves.png、{tag}_confusion.png")
    except Exception as e:
        print(f"(画图跳过：{e})")

    print("\nDay 6 完成 ✅")


if __name__ == "__main__":
    main()
