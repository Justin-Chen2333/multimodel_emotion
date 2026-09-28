"""
Day 5 — MELD 文本 Dataset + DataLoader + RoBERTa 训练循环（1 epoch）

用法（Anaconda Prompt，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python text_day5.py --limit 320          # 冒烟测试：只用 320 条 train / 320 条 dev，几十秒跑完
    python text_day5.py                      # 今日最低目标：全量 train 1 epoch，每 10 个 batch 打印 loss
    python text_day5.py --log-every 1        # 严格按 checklist：每个 batch 都打印
    python text_day5.py --wandb              # 可选：loss 曲线记录到 wandb（先 python -m pip install wandb && wandb login）
    python text_day5.py --save               # 另存权重 results\\text_day5.pt（约 500MB，Day 6 才正式存 best）

按 Day 4 交接的设定：
  - TextClassifier / load_split / LABELS 直接从 roberta_day4 import，label 顺序与 results\\label_map.json 核对
  - max_length 128，只截断不 pad；collate 时 pad 到本 batch 最长句（动态 padding）
  - AdamW lr 2e-5，weight decay 0.01（bias/LayerNorm 不加），前 10% 步线性 warmup 再线性衰减，梯度裁剪 1.0
  - 不加 class weights（Day 6 再对比 不加权 / balanced / sqrt(balanced)）
  - 4090 上用 bf16 autocast 加速（--no-amp 关掉）
输出：
  results\\day5_train_log.csv   每个 step 的 loss 和 lr
  results\\day5_loss.png        loss 曲线
"""
import argparse
import csv
import json
import math
import os
import random
import time
from functools import partial

# 必须在 import transformers 之前设置
os.environ.setdefault("HF_HOME", r"F:\hf_cache")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from roberta_day4 import LABELS, MODEL_NAME, OUT_DIR, TextClassifier, load_split


# ---------------------------------------------------------------- Dataset
class TextDataset(Dataset):
    """一次性把所有句子 tokenize 好（不 pad），__getitem__ 返回 (input_ids, label)。"""

    def __init__(self, df, tokenizer, max_length=128):
        enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.input_ids = enc["input_ids"]
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.labels[i]


def collate(batch, pad_id):
    """动态 padding：pad 到本 batch 最长句，同时生成 attention_mask。"""
    L = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    return input_ids, mask, labels


# ---------------------------------------------------------------- 评估
def f1_report(pred, gold, k=len(LABELS)):
    f1s, sup = [], []
    for c in range(k):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        sup.append(int((gold == c).sum()))
    return float(np.average(f1s, weights=sup)), f1s, sup


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    preds, golds, loss_sum = [], [], 0.0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(ids, mask)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        preds.append(logits.argmax(-1).cpu())
        golds.append(y.cpu())
    p, g = torch.cat(preds).numpy(), torch.cat(golds).numpy()
    wf1, f1s, sup = f1_report(p, g)
    return dict(loss=loss_sum / len(g), acc=float((p == g).mean()), wf1=wf1, f1s=f1s, sup=sup,
                pred_counts=np.bincount(p, minlength=len(LABELS)).tolist(), gold=g)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1, help="warmup 占总步数的比例")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟测试）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.no_amp
    print(f"device={device}  bf16={amp}  HF_HOME={os.environ['HF_HOME']}")

    # label 顺序核对
    lm = OUT_DIR / "label_map.json"
    if lm.exists():
        saved = json.loads(lm.read_text(encoding="utf-8"))["labels"]
        assert saved == LABELS, f"label 顺序和 label_map.json 不一致：{saved} vs {LABELS}"
        print(f"label 顺序与 {lm.name} 一致：{LABELS}")

    # ---------- 1. Dataset ----------
    print("\n========== 1. TextDataset ==========")
    train_df, dev_df = load_split("train"), load_split("dev")
    if a.limit:
        train_df = train_df.sample(n=min(a.limit, len(train_df)), random_state=a.seed)
        dev_df = dev_df.head(a.limit)
    tok = RobertaTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    train_ds = TextDataset(train_df, tok, a.max_length)
    dev_ds = TextDataset(dev_df, tok, a.max_length)
    print(f"train {len(train_ds)} 条，dev {len(dev_ds)} 条（tokenize 用时 {time.time() - t0:.1f}s）")
    ids0, y0 = train_ds[0]
    print(f"train_ds[0] → input_ids 长度 {len(ids0)}，label {y0} ({LABELS[y0]})")
    print(f"  文本：{tok.decode(ids0, skip_special_tokens=True)!r}")

    # ---------- 2. DataLoader ----------
    print("\n========== 2. DataLoader ==========")
    col = partial(collate, pad_id=tok.pad_token_id)
    g = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, collate_fn=col, generator=g)
    dev_dl = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=col)
    n_batches = 0
    for ids, mask, y in train_dl:  # 完整遍历一遍，确认不报错
        n_batches += 1
    print(f"train 遍历完毕：{n_batches} 个 batch；最后一个 batch input_ids {tuple(ids.shape)}，labels {tuple(y.shape)}")

    # ---------- 3. 模型 + 优化器 ----------
    print("\n========== 3. 模型 / AdamW / warmup ==========")
    model = TextClassifier().to(device)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": a.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    total = len(train_dl) * a.epochs
    warm = int(total * a.warmup)
    sched = get_linear_schedule_with_warmup(opt, warm, total)
    print(f"总步数 {total}，warmup {warm} 步，峰值 lr {a.lr}")

    if a.wandb:
        import wandb
        wandb.init(project="emotion-ai", name=f"day5-text-bs{a.batch}-lr{a.lr}", config=vars(a))

    # ---------- 4. 训练 ----------
    print("\n========== 4. 训练 ==========")
    log, step, t_start = [], 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for i, (ids, mask, y) in enumerate(train_dl, 1):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(ids, mask)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step()
            sched.step()
            step += 1

            l = loss.item()
            run += l
            lr = sched.get_last_lr()[0]
            log.append(dict(epoch=ep, step=step, loss=round(l, 5), lr=lr, grad_norm=round(gnorm, 4)))
            if a.wandb:
                wandb.log({"train_loss": l, "lr": lr, "grad_norm": gnorm}, step=step)
            if step == 1:
                print(f"第 1 步 loss {l:.4f}（ln7 = {math.log(7):.4f}，接近即正常）")
            if i % a.log_every == 0 or i == len(train_dl):
                print(f"epoch {ep}  batch {i:4d}/{len(train_dl)}  loss {l:.4f}  "
                      f"平均 {run / i:.4f}  lr {lr:.2e}  |grad| {gnorm:.2f}")

        # ---------- 5. dev 快速评估 ----------
        r = evaluate(model, dev_dl, device, amp)
        base_wf1, _, _ = f1_report(np.full_like(r["gold"], LABELS.index("neutral")), r["gold"])
        print(f"\n[epoch {ep}] dev loss {r['loss']:.4f}  acc {r['acc']:.4f}  weighted F1 {r['wf1']:.4f}"
              f"   (全猜 neutral 的 weighted F1 = {base_wf1:.4f})")
        print("  各类 F1 / 支持数 / 预测数：")
        for c, name in enumerate(LABELS):
            print(f"    {name:9s} F1 {r['f1s'][c]:.3f}   gold {r['sup'][c]:4d}   pred {r['pred_counts'][c]:4d}")
        if a.wandb:
            wandb.log({"dev_loss": r["loss"], "dev_acc": r["acc"], "dev_wf1": r["wf1"]}, step=step)

    mins = (time.time() - t_start) / 60
    print(f"\n训练用时 {mins:.1f} 分钟")
    if device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ---------- 6. 保存 ----------
    OUT_DIR.mkdir(exist_ok=True)
    log_path = OUT_DIR / "day5_train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    k = min(50, len(log) // 2) or 1
    first = np.mean([x["loss"] for x in log[:k]])
    last = np.mean([x["loss"] for x in log[-k:]])
    print(f"前 {k} 步平均 loss {first:.4f} → 最后 {k} 步平均 {last:.4f}"
          f"（{'下降 ✅' if last < first else '没有下降 ⚠️'}）")
    print(f"已保存 {log_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        losses = np.array([x["loss"] for x in log])
        win = max(1, min(25, len(losses) // 10))
        smooth = np.convolve(losses, np.ones(win) / win, mode="valid")
        plt.figure(figsize=(8, 4))
        plt.plot(losses, alpha=0.3, label="batch loss")
        plt.plot(np.arange(win - 1, len(losses)), smooth, label=f"moving avg ({win})")
        plt.axhline(math.log(7), ls="--", c="gray", lw=0.8, label="ln 7")
        plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.title("Day 5 — RoBERTa text-only, train loss")
        plt.legend(); plt.tight_layout()
        png = OUT_DIR / "day5_loss.png"
        plt.savefig(png, dpi=120)
        print(f"已保存 {png}")
    except Exception as e:
        print(f"(画图跳过：{e})")

    if a.save:
        pt = OUT_DIR / "text_day5.pt"
        torch.save(model.state_dict(), pt)
        print(f"已保存 {pt}")
    if a.wandb:
        wandb.finish()
    print("\nDay 5 完成 ✅")


if __name__ == "__main__":
    main()
