"""
Day 12 — 多模态 Late Fusion 正式训练：5 epochs × 3 seeds，记录 dev weighted F1，和文本基线对比

用法（PowerShell，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python src\\fusion_day12.py --limit 320 --epochs 2 --seeds 42 43   # 冒烟测试（1 分钟内），输出写 day12_smoke_*，不存权重
    python src\\fusion_day12.py                                         # 今日最低目标：5 epochs × seed 42/43/44（约 7 分钟）
    python src\\fusion_day12.py --wandb                                 # 同上，每个 seed 一条 wandb run（group=day12）
    python src\\fusion_day12.py --freeze-text --tag day12_frozen        # 可选（计划里的"冻结 RoBERTa 只训 MLP"）：另存一套输出，不覆盖 day12_*

固定配置（Day 11 选定，今天不再调）：
  LateFusionClassifier(("text","audio"), text_norm=True)；RoBERTa 一起微调 lr 2e-5；层权重 / LN / 融合头 lr 1e-3；
  AdamW wd 0.01；10% warmup + 线性衰减（按 5 个 epoch 的总步数）；梯度裁剪 1.0；bf16；batch 16；不加 class weights
训练本身全部复用 fusion_day11.train_one（传 out={} → 记住最佳 epoch 的权重，打乱测试在最佳 epoch 上做）。
今天新加的是"怎么看结果"：
  1. 每个 seed 一张按 epoch 的表：train loss（这个 epoch 所有步的平均）/ dev loss / acc / wF1 / mF1 → 看 dev loss 从哪个 epoch 开始回升
  2. 每个 seed 按 dev wF1 取最佳 epoch；3 个 seed 汇总成 均值 ± 标准差，和文本基线 3 个 seed 并排（同样是"每个 seed 取最佳 epoch"）
  3. 各类 F1：多模态（3 seed 平均）vs 文本基线（seed 42）→ 哪些情绪类别从音频里得到了帮助
  4. 配对 bootstrap：多模态 seed 42 vs 文本基线 seed 42，在同一批 1108 条 dev 上重采样 → wF1 差值的 95% 置信区间
  5. 打乱测试（最佳 epoch）、去均值贡献、13 层权重，3 个 seed 平均
  6. 只保存预先定好的 seed（--save-seed，默认 42）的最佳 epoch → results\\fusion_best.pt + fusion_best.json
     （和 Day 7 定的 text_best 规则一样：按事先定好的配置 + seed 存，不在多次运行里挑 dev 最高分）
只用 dev（1108 条），test 不碰（Day 14 才用）。
输出（tag 默认 day12；冒烟测试为 day12_smoke）：
  results\\{tag}_epochs.csv           每个 seed 每个 epoch 一行
  results\\{tag}_runs.csv             每个 seed 一行（最佳 epoch 的指标、各类 F1、打乱测试、层权重、去掉 shared_audio 的 wF1）
  results\\{tag}_steps.csv            每一步的 loss / 梯度 / lr
  results\\{tag}_dev_probs_s{seed}.npy  每个 seed 最佳 epoch 的 dev 概率 (1108, 7)，行顺序 = build_table("dev") 的顺序
                                       → Day 13/14 画混淆矩阵、做错误分析直接读
  results\\{tag}_curves.png           左：loss 随 epoch；中：dev wF1 随 epoch（带文本基线区间）；右：各类 F1 对比
  results\\{tag}_summary.json         以上所有汇总 + 配置
  results\\fusion_best.pt / .json      seed 42 最佳 epoch 的权重（纯 state_dict，约 500MB，被 .gitignore 排除）
"""
import argparse
import json
import math
import os
import time
from functools import partial

# 必须在 import transformers 之前设置
os.environ.setdefault("HF_HOME", r"F:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from fusion_day10 import AUDIO_MODEL, MultimodalDataset, build_table, load_shared_audio
from fusion_day11 import SCALE_FIX, load_reference, run_name, train_one
from roberta_day4 import LABELS, MODEL_NAME, OUT_DIR, TextClassifier, load_split
from text_day6 import metrics, predict

try:  # 和 text_day6 一样兼容本机旧文件名
    from epoch1_day5 import TextDataset, collate
except ImportError:
    from text_day5 import TextDataset, collate

K = len(LABELS)
# 文本基线 3 个 seed（Day 6：不加权，每个 seed 取 dev wF1 最高的 epoch）。Day 6 是 dev 1109 口径；
# Day 10 验证过去掉缺 wav 的 1 条只让 wF1 变 0.0003，所以直接引用，表注里说明
TEXT3 = dict(seeds=[42, 43, 44], wf1=[0.6068, 0.6030, 0.6023], acc_mean=0.618, acc_sd=0.003,
             mf1_mean=0.458, mf1_sd=0.003, best_epochs=[2, 3, 3],
             note="Day 6 · 不加权 · 每个 seed 取最佳 epoch · dev 1109 口径（和 1108 口径差约 0.0003）")
# Day 11 定下的配置；只有和它完全一样时才写 fusion_best.pt，否则另起名字，避免变体实验覆盖正式权重
DEFAULT_CFG = dict(modalities=("text", "audio"), scale_fix="text_ln", lr_text=2e-5, lr_head=1e-3, freeze_text=False, epochs=5)


# ================================================================ 文本基线在同一批 1108 条上的概率
def text_baseline_probs(dev_df, tok, device, amp, max_length):
    """→ (probs (1108, 7) 按 dev_df 的行顺序, 来源说明) 或 (None, 原因)。
    优先读 Day 6 存的 day6_dev_probs.npy（none / seed 42，Day 7 恢复过），按 row_csv 取出 1108 行；
    先和 text_best.json 的分数核对，对不上就用 text_best.pt 重新推理。"""
    rows = dev_df["row_csv"].to_numpy()
    gold_full = load_split("dev")["label"].to_numpy()
    meta_p = OUT_DIR / "text_best.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    p = OUT_DIR / "day6_dev_probs.npy"
    if p.exists():
        probs = np.load(p)
        if len(probs) == len(gold_full):
            wf1 = metrics(probs, gold_full)["wf1"]
            if not meta or abs(wf1 - meta["dev_wf1"]) < 2e-3:
                return probs[rows], f"day6_dev_probs.npy（1109 条 wF1 {wf1:.4f}，和 text_best.json 一致）"
            print(f"  ⚠️ day6_dev_probs.npy 的 wF1 {wf1:.4f} 和 text_best.json 的 {meta['dev_wf1']} 对不上（被别的运行覆盖过？），"
                  f"改用 text_best.pt 重新推理")
    pt = OUT_DIR / "text_best.pt"
    if not pt.exists():
        return None, "找不到 day6_dev_probs.npy 和 text_best.pt"
    tm = TextClassifier().to(device)
    tm.load_state_dict(torch.load(pt, map_location=device))
    dev_full = load_split("dev").reset_index(drop=True)
    dl = DataLoader(TextDataset(dev_full, tok, max_length), batch_size=64, shuffle=False,
                    collate_fn=partial(collate, pad_id=tok.pad_token_id))
    probs, _ = predict(tm, dl, device, amp)
    del tm
    if device == "cuda":
        torch.cuda.empty_cache()
    return probs[rows], "text_best.pt 重新推理"


def paired_bootstrap(gold, pa, pb, n=2000, seed=0):
    """配对 bootstrap：每次从 N 条 dev 里有放回地抽 N 条（两个模型用同一组下标），算 wF1(a) − wF1(b)。
    → 观测差值、95% 区间（2.5% / 97.5% 分位）、差值 ≤ 0 的比例。
    它回答的是\"换一批同分布的 dev 句子，这个差值还稳不稳\"——只覆盖数据抽样的噪声，不覆盖训练 seed 的噪声"""
    rng = np.random.default_rng(seed)
    ya, yb, N = pa.argmax(1), pb.argmax(1), len(gold)
    lab = list(range(K))

    def wf1(y, p):
        return f1_score(y, p, labels=lab, average="weighted", zero_division=0)

    obs = wf1(gold, ya) - wf1(gold, yb)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        diffs[i] = wf1(gold[idx], ya[idx]) - wf1(gold[idx], yb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(diff=float(obs), lo=float(lo), hi=float(hi), p_le0=float((diffs <= 0).mean()), n=n)


def ms(x, d=4):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--save-seed", type=int, default=42, help="只保存这个 seed 的最佳 epoch 权重（事先定好，不挑最高分）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--modalities", nargs="+", default=["text", "audio"], choices=["text", "audio"])
    ap.add_argument("--scale-fix", default="text_ln", choices=list(SCALE_FIX))
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-text", type=float, default=2e-5)
    ap.add_argument("--freeze-text", action="store_true", help="冻结 RoBERTa，只训融合部分（对照）")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=0, help="每 N 步额外评估一次 dev（默认 0 = 只在 epoch 末）")
    ap.add_argument("--audio-model", default=AUDIO_MODEL)
    ap.add_argument("--tag", default="day12", help="输出文件前缀；跑变体实验时换一个，避免覆盖 day12_*")
    ap.add_argument("--bootstrap", type=int, default=2000, help="配对 bootstrap 次数（0 = 跳过）")
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试：train 随机 N 条、dev 前 N 条（不存权重）")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    if a.limit:
        a.tag = f"{a.tag}_smoke"
    tag = a.tag

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.no_amp
    print(f"device={device}  bf16={amp}  HF_HOME={os.environ['HF_HOME']}  audio={a.audio_model}  tag={tag}")
    OUT_DIR.mkdir(exist_ok=True)
    lm = OUT_DIR / "label_map.json"
    if lm.exists():
        saved = json.loads(lm.read_text(encoding="utf-8"))["labels"]
        assert saved == LABELS, f"label 顺序和 label_map.json 不一致：{saved} vs {LABELS}"
    ref = load_reference()

    # ---------- 数据（只用 train / dev）----------
    print("\n========== 数据 ==========")
    dups = load_shared_audio()
    train_df, train_emb, _ = build_table("train", a.audio_model, dups)
    dev_df, dev_emb, _ = build_table("dev", a.audio_model, dups)
    if a.limit:
        train_df = train_df.sample(n=min(a.limit, len(train_df)), random_state=42).reset_index(drop=True)
        dev_df = dev_df.head(a.limit).reset_index(drop=True)
        print(f"  冒烟测试：train {len(train_df)} / dev {len(dev_df)} 条")
    tok = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MultimodalDataset(train_df, train_emb, tok, a.max_length)
    dev_ds = MultimodalDataset(dev_df, dev_emb, tok, a.max_length)
    del train_emb, dev_emb
    spe = math.ceil(len(train_ds) / a.batch)                       # steps per epoch
    gold = np.asarray(dev_ds.labels)
    shared = np.asarray(dev_ds.shared_audio, dtype=bool)
    support = np.bincount(gold, minlength=K)
    print(f"  train {len(train_ds)} / dev {len(dev_ds)}；每 epoch {spe} 步，{a.epochs} epochs 共 {spe * a.epochs} 步")

    # ---------- 文本基线（同一批 dev 句子）----------
    print("\n========== 文本基线（对比对象）==========")
    text_probs, text_src = text_baseline_probs(dev_df, tok, device, amp, a.max_length)
    text_m = metrics(text_probs, gold) if text_probs is not None else None
    t_wf1 = np.array(TEXT3["wf1"])
    print(f"  3 个 seed（{TEXT3['note']}）：wF1 {ms(t_wf1)}，mF1 {TEXT3['mf1_mean']:.3f} ± {TEXT3['mf1_sd']:.3f}，"
          f"acc {TEXT3['acc_mean']:.3f} ± {TEXT3['acc_sd']:.3f}，最佳 epoch {TEXT3['best_epochs']}")
    if text_m:
        print(f"  seed 42 在这 {len(gold)} 条上：acc {text_m['acc']:.4f}  wF1 {text_m['wf1']:.4f}  mF1 {text_m['mf1']:.4f}"
              f"（来源：{text_src}）")
    else:
        print(f"  ⚠️ 拿不到文本基线的逐条概率（{text_src}）→ 跳过各类 F1 对比和 bootstrap，只比 3 seed 的汇总数")
    print(f"  全猜 neutral：wF1 {ref['neutral']['wf1']:.4f}")

    # ---------- 训练 ----------
    base_cfg = dict(modalities=tuple(m for m in ("text", "audio") if m in a.modalities), scale_fix=a.scale_fix,
                    lr_text=a.lr_text, lr_head=a.lr_head, freeze_text=a.freeze_text, epochs=a.epochs)
    is_default = base_cfg == DEFAULT_CFG
    save_name = "fusion_best" if is_default else f"fusion_{tag}"
    print(f"\n共 {len(a.seeds)} 次运行（seeds {a.seeds}），每次 {a.epochs} epochs；配置 "
          f"{'= Day 11 选定的正式配置' if is_default else '≠ 正式配置（变体实验）'}，"
          + (f"seed {a.save_seed} 的最佳 epoch 存为 {save_name}.pt" if not (a.limit or a.no_save) else "不保存权重（--limit / --no-save）"))

    runs, epoch_rows, all_steps, mm_probs, shuf_probs = [], [], [], {}, {}
    t_all = time.time()
    for j, seed in enumerate(a.seeds, 1):
        cfg = dict(base_cfg, seed=seed)
        name = run_name(cfg)
        print(f"\n========== {j}/{len(a.seeds)}  {name} ==========")
        out = {}
        s, st, ev = train_one(cfg, train_ds, dev_ds, a, device, amp, tok.pad_token_id, out=out)
        all_steps += st

        # 每个 epoch 一行：train loss = 这个 epoch 所有步的平均；dev 指标取 epoch 末那次评估
        st_df, ev_df = pd.DataFrame(st), pd.DataFrame(ev)
        tl = st_df.groupby("epoch").loss.mean()
        ev_end = ev_df[(ev_df.step > 0) & (ev_df.step % spe == 0)].copy()
        ev_end["epoch"] = (ev_end.step // spe).astype(int)
        for r in ev_end.itertuples():
            epoch_rows.append(dict(seed=seed, epoch=r.epoch, train_loss=round(float(tl[r.epoch]), 5),
                                   dev_loss=r.dev_loss, dev_acc=r.dev_acc, dev_wf1=r.dev_wf1, dev_mf1=r.dev_mf1,
                                   **{f"f1_{l}": getattr(r, f"f1_{l}") for l in LABELS},
                                   **{c: getattr(r, c) for c in ev_df.columns if c.startswith(("norm_", "contrib"))}))

        probs = out["probs"]
        assert (out["gold"] == gold).all()
        mm_probs[seed] = probs
        shuf_probs[seed] = out["shuffled_probs"]
        np.save(OUT_DIR / f"{tag}_dev_probs_s{seed}.npy", probs)
        m_ns = metrics(probs[~shared], gold[~shared])               # 去掉 shared_audio 的 2 条再算
        s.update(dev_wf1_no_shared=round(m_ns["wf1"], 4), n_no_shared=int((~shared).sum()),
                 pred_counts=" ".join(str(x) for x in np.bincount(probs.argmax(1), minlength=K)))
        runs.append(s)

        if seed == a.save_seed and not a.limit and not a.no_save:
            pt = OUT_DIR / f"{save_name}.pt"
            torch.save(out["state"], pt)
            meta = dict(config={**cfg, "modalities": list(cfg["modalities"])}, model_kwargs=SCALE_FIX[cfg["scale_fix"]],
                        best_epoch=s["best_epoch"], dev_acc=s["dev_acc"], dev_wf1=s["dev_wf1"], dev_mf1=s["dev_mf1"],
                        dev_loss=s["dev_loss"], per_class_f1={l: s[f"f1_{l}"] for l in LABELS},
                        layer_weights=[float(x) for x in s.get("layer_weights", "").split()],
                        n_dev=len(gold), batch=a.batch, warmup=a.warmup, audio_model=a.audio_model,
                        text_model=MODEL_NAME, labels=LABELS,
                        load_hint=f"LateFusionClassifier({tuple(cfg['modalities'])}, **{SCALE_FIX[cfg['scale_fix']]})"
                                  f".load_state_dict(torch.load('{pt.name}'))",
                        saved_at=time.strftime("%Y-%m-%d %H:%M"))
            (OUT_DIR / f"{save_name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  已保存 {pt.name}（{pt.stat().st_size / 1e6:.0f} MB）+ {save_name}.json"
                  f"（seed {seed}，最佳 epoch {s['best_epoch']}，dev wF1 {s['dev_wf1']:.4f}）")
        del out
    mins = (time.time() - t_all) / 60
    print(f"\n全部训练用时 {mins:.1f} 分钟")
    gpu_peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None
    if gpu_peak:
        print(f"显存峰值 {gpu_peak} GB")

    runs_df, ep_df = pd.DataFrame(runs), pd.DataFrame(epoch_rows)
    runs_df.to_csv(OUT_DIR / f"{tag}_runs.csv", index=False)
    ep_df.to_csv(OUT_DIR / f"{tag}_epochs.csv", index=False)
    pd.DataFrame(all_steps).to_csv(OUT_DIR / f"{tag}_steps.csv", index=False)

    # ================================================================ 1. 按 epoch 的表
    print("\n========== 1. 每个 seed 按 epoch（★ = dev wF1 最高，◆ = dev loss 最低）==========")
    for seed, g in ep_df.groupby("seed", sort=False):
        bw, bl = g.dev_wf1.idxmax(), g.dev_loss.idxmin()
        print(f"  seed {seed}")
        print(f"    {'epoch':>5} {'train loss':>10} {'dev loss':>8} {'dev acc':>7} {'dev wF1':>7} {'dev mF1':>7}")
        for i, r in g.iterrows():
            print(f"    {int(r.epoch):>5} {r.train_loss:>10.4f} {r.dev_loss:>8.4f} {r.dev_acc:>7.4f} {r.dev_wf1:>7.4f} "
                  f"{r.dev_mf1:>7.4f}  {'★' if i == bw else ' '}{'◆' if i == bl else ''}")
    loss_min_ep = ep_df.loc[ep_df.groupby("seed").dev_loss.idxmin()].epoch.tolist()
    print(f"  dev loss 最低的 epoch：{loss_min_ep}；dev wF1 最高的 epoch：{runs_df.best_epoch.tolist()}"
          f"（文本基线是 {TEXT3['best_epochs']}）")

    # ================================================================ 2. 汇总：多模态 vs 文本
    mm_w = runs_df.dev_wf1.to_numpy()
    delta = mm_w.mean() - t_wf1.mean()
    print("\n========== 2. 汇总（每个 seed 取最佳 epoch；均值 ± 标准差）==========")
    print(f"  {'模型':22s} {'n':>2} {'dev acc':>17} {'dev weighted F1':>17} {'dev macro F1':>17}  best epochs")
    print(f"  {'text-only（Day 6）':22s} {3:>2} {TEXT3['acc_mean']:>9.3f} ± {TEXT3['acc_sd']:.3f}  {ms(t_wf1):>17} "
          f"{TEXT3['mf1_mean']:>9.3f} ± {TEXT3['mf1_sd']:.3f}  {TEXT3['best_epochs']}")
    mm_label = "+".join(base_cfg["modalities"]) + ("（冻结 RoBERTa）" if a.freeze_text else "")
    print(f"  {mm_label:22s} {len(runs_df):>2} {ms(runs_df.dev_acc):>17} {ms(mm_w):>17} {ms(runs_df.dev_mf1):>17}  "
          f"{runs_df.best_epoch.tolist()}")
    print(f"  各 seed 的 wF1：" + "  ".join(f"s{r.seed} {r.dev_wf1:.4f}" for r in runs_df.itertuples())
          + f"   ｜ 文本：" + "  ".join(f"s{s_} {w_:.4f}" for s_, w_ in zip(TEXT3["seeds"], t_wf1)))
    print(f"  Δ wF1（多模态均值 − 文本均值）= {delta:+.4f}；Δ mF1 = {runs_df.dev_mf1.mean() - TEXT3['mf1_mean']:+.4f}")
    if len(mm_w) > 1:
        pooled = math.sqrt((mm_w.var(ddof=1) + t_wf1.var(ddof=1)) / 2)
        print(f"  两边 seed 之间的标准差：多模态 {mm_w.std(ddof=1):.4f}，文本 {t_wf1.std(ddof=1):.4f}"
              f"（合并 {pooled:.4f}）→ Δ 约是 {abs(delta) / max(pooled, 1e-9):.1f} 个标准差")
    print(f"  去掉 shared_audio 的 dev 条目后（n={runs_df.n_no_shared.iloc[0]}）：多模态 wF1 {ms(runs_df.dev_wf1_no_shared)}")

    # ================================================================ 3. 各类 F1
    print(f"\n========== 3. 各类 F1：文本基线 seed 42 vs 多模态（{len(runs_df)} 个 seed 平均）==========")
    mm_pc = runs_df[[f"f1_{l}" for l in LABELS]].to_numpy()        # (n_seeds, 7)
    per_class = []
    print(f"  {'类别':9s} {'dev 条数':>8} {'文本 s42':>9} {'多模态 均值 ± sd':>18} {'Δ':>7}")
    for i, l in enumerate(LABELS):
        t_ = text_m["per_class"][i] if text_m else float("nan")
        row = dict(label=l, support=int(support[i]), text_s42=round(t_, 4), mm_mean=round(float(mm_pc[:, i].mean()), 4),
                   mm_sd=round(float(mm_pc[:, i].std(ddof=1)), 4) if len(mm_pc) > 1 else 0.0)
        row["delta"] = round(row["mm_mean"] - t_, 4)
        per_class.append(row)
        print(f"  {l:9s} {support[i]:>8d} {t_:>9.3f} {row['mm_mean']:>10.3f} ± {row['mm_sd']:.3f} {row['delta']:>+7.3f}")
    if text_m:
        up = sorted(per_class, key=lambda r: -r["delta"])
        print(f"  提升最多：{up[0]['label']} {up[0]['delta']:+.3f}、{up[1]['label']} {up[1]['delta']:+.3f}；"
              f"下降最多：{up[-1]['label']} {up[-1]['delta']:+.3f}")
        small = [f"{r['label']} {r['support']} 条" for r in per_class if r["support"] < 50]
        if small:
            print(f"  （{'、'.join(small)}：样本很少，F1 差 0.1 可能只是多对 / 少对两三条，别过度解读）")
    pred_cnt = np.array([[int(x) for x in s_.split()] for s_ in runs_df.pred_counts])
    print("  预测条数（多模态 seed 平均 / 文本 s42 / 真实）：" + "  ".join(
        f"{l} {pred_cnt[:, i].mean():.0f}/{(np.bincount(text_m['pred'], minlength=K)[i] if text_m else 0)}/{support[i]}"
        for i, l in enumerate(LABELS)))

    # ================================================================ 4. 配对 bootstrap
    boot = None
    if text_m and a.save_seed in mm_probs and a.bootstrap:
        print(f"\n========== 4. 配对 bootstrap（多模态 seed {a.save_seed} vs 文本 seed 42，同一批 {len(gold)} 条，"
              f"{a.bootstrap} 次）==========")
        t0 = time.time()
        boot = paired_bootstrap(gold, mm_probs[a.save_seed], text_probs, n=a.bootstrap)
        print(f"  wF1 差值 {boot['diff']:+.4f}，95% 区间 [{boot['lo']:+.4f}, {boot['hi']:+.4f}]，"
              f"重采样里差值 ≤ 0 的比例 {boot['p_le0']:.3f}（{time.time() - t0:.0f}s）")
        inside = boot["lo"] <= 0 <= boot["hi"]
        print(f"  区间{'含 0 → 差值在 dev 抽样噪声范围内（换一批同分布的句子，正负都可能）' if inside else '不含 0 → 这个差值不太像只是碰巧抽到这批 dev 句子'}。"
              "它只比较 seed 42 这一对模型，seed 之间的噪声看第 2 节")

    # ================================================================ 5. 模态依赖
    print(f"\n========== 5. 模态依赖（最佳 epoch；{len(runs_df)} 个 seed 平均）==========")
    if "shuffled_audio_wf1" in runs_df:
        sa = runs_df.shuffled_audio_wf1 - runs_df.dev_wf1
        stt = runs_df.shuffled_text_wf1 - runs_df.dev_wf1
        print(f"  打乱音频 wF1 变化 {ms(sa)}（各 seed：{', '.join(f'{x:+.4f}' for x in sa)}）")
        print(f"  打乱文本 wF1 变化 {ms(stt)}")
        print(f"  Day 11（1 epoch，seed 42）是 打乱音频 −0.016 / 打乱文本 −0.342，可以对照看多训几个 epoch 后音频还被用多少")
    for m_ in base_cfg["modalities"]:
        if f"contribc_{m_}" in runs_df:
            print(f"  {m_:5s}：范数 {ms(runs_df[f'norm_{m_}'], 1)}，贡献 {ms(runs_df[f'contrib_{m_}'], 1)}，"
                  f"去均值贡献 {ms(runs_df[f'contribc_{m_}'], 1)}")
    lw = None
    if "layer_weights" in runs_df:
        lw = np.array([[float(x) for x in s_.split()] for s_ in runs_df.layer_weights]).mean(0)
        print("  13 层权重（seed 平均）：" + " ".join(f"{x:.3f}" for x in lw)
              + f"  → 最大第 {int(lw.argmax())} 层，最小第 {int(lw.argmin())} 层（均匀 = {1 / 13:.3f}）")

    # ================================================================ 结论 + checklist
    print("\n========== 结论 ==========")
    # 两种噪声分开看：seed 之间（第 2 节，|Δ| 是否超过 2 个合并标准差）和 dev 抽样（第 4 节，bootstrap 区间是否含 0）
    pooled = math.sqrt((mm_w.var(ddof=1) + t_wf1.var(ddof=1)) / 2) if len(mm_w) > 1 else float("nan")
    seed_sig = bool(len(mm_w) > 1 and abs(delta) > 2 * pooled)
    boot_sig = boot is not None and not (boot["lo"] <= 0 <= boot["hi"]) and np.sign(boot["diff"]) == np.sign(delta)
    word = "高" if delta > 0 else "低"
    if abs(delta) < 0.005 and not seed_sig and not boot_sig:
        verdict = f"多模态和文本基线持平（Δ {delta:+.4f}）"
    elif seed_sig and (boot is None or boot_sig):
        verdict = f"多模态比文本基线{word} {abs(delta):.4f} wF1，seed 之间和 dev 抽样两种噪声下都站得住"
    elif seed_sig or boot_sig:
        verdict = (f"多模态比文本基线{word} {abs(delta):.4f} wF1，但只有 "
                   f"{'seed 之间的比较' if seed_sig else 'bootstrap'}支持，"
                   f"{'bootstrap 区间含 0（dev 句子太少，单对模型分不开）' if seed_sig else 'seed 之间的波动比差值还大'}")
    else:
        verdict = f"多模态均值{word} {abs(delta):.4f} wF1，但在噪声范围内（seed 之间和 bootstrap 都分不开）"
    noisy = not (seed_sig and (boot is None or boot_sig))
    print(f"  {verdict}")
    if delta <= 0 or noisy:
        print("  按 Day 11 交接的预案（先看完结果再决定，不要边跑边改）：① modality dropout（p≈0.3 把文本一路置零）"
              "② 融合头 dropout 0.3 ③ RoBERTa lr 1e-5。换 --tag 跑，不覆盖 day12_*")
    checks = {
        f"{len(a.seeds)} 个 seed × {a.epochs} epochs 多模态训练完成": all(r["steps"] == spe * a.epochs for r in runs),
        f"记录到 val weighted F1：{ms(mm_w)}": bool(np.isfinite(mm_w).all()),
        f"vs 文本基线的变化已记录：Δ {delta:+.4f}": bool(np.isfinite(delta)),
    }
    print("\nDay 12 checklist：")
    for k_, v in checks.items():
        print(f"  {'✅' if v else '⚠️'} {k_}")

    summary = dict(args={k_: v for k_, v in vars(a).items()}, config=dict(base_cfg, modalities=list(base_cfg["modalities"])),
                   is_default_config=is_default, reference=ref, text3=TEXT3,
                   text_s42_dev1108=(dict(source=text_src, acc=round(text_m["acc"], 4), wf1=round(text_m["wf1"], 4),
                                          mf1=round(text_m["mf1"], 4)) if text_m else None),
                   multimodal=dict(wf1=[float(x) for x in mm_w], wf1_mean=round(float(mm_w.mean()), 4),
                                   wf1_sd=round(float(mm_w.std(ddof=1)), 4) if len(mm_w) > 1 else None,
                                   acc_mean=round(float(runs_df.dev_acc.mean()), 4),
                                   mf1_mean=round(float(runs_df.dev_mf1.mean()), 4),
                                   best_epochs=runs_df.best_epoch.tolist(), dev_loss_min_epochs=loss_min_ep),
                   delta_wf1=round(float(delta), 4), per_class=per_class, bootstrap=boot,
                   layer_weights_mean=[round(float(x), 4) for x in lw] if lw is not None else None,
                   verdict=verdict, checks={k_: bool(v) for k_, v in checks.items()}, runs=runs,
                   minutes=round(mins, 1), gpu_peak_GB=gpu_peak, saved_at=time.strftime("%Y-%m-%d %H:%M"))
    (OUT_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                                  encoding="utf-8")

    # ================================================================ 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
        for c, (seed, g) in zip(colors, ep_df.groupby("seed", sort=False)):
            ax[0].plot(g.epoch, g.train_loss, marker="o", c=c, label=f"train s{seed}")
            ax[0].plot(g.epoch, g.dev_loss, marker="s", ls="--", c=c, label=f"dev s{seed}")
            ax[1].plot(g.epoch, g.dev_wf1, marker="o", c=c, label=f"text+audio s{seed}")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("loss per epoch (solid: train, dashed: dev)")
        ax[1].axhspan(t_wf1.mean() - t_wf1.std(ddof=1), t_wf1.mean() + t_wf1.std(ddof=1), color="tab:red", alpha=0.15,
                      label=f"text-only, 3 seeds: {t_wf1.mean():.3f} ± {t_wf1.std(ddof=1):.3f}")
        ax[1].axhline(t_wf1.mean(), c="tab:red", lw=1)
        ax[1].axhline(ref["neutral"]["wf1"], ls=":", c="gray", lw=1, label=f"all-neutral {ref['neutral']['wf1']:.3f}")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("dev weighted F1"); ax[1].set_title(f"dev weighted F1 ({len(gold)} utts)")
        lo_y = min(ep_df.dev_wf1.min(), t_wf1.min()) - 0.03
        ax[1].set_ylim(max(0, lo_y), max(ep_df.dev_wf1.max(), t_wf1.max()) + 0.02)
        for x in ax[:2]:
            x.set_xticks(range(1, a.epochs + 1)); x.legend(fontsize=7)
        xs = np.arange(K)
        if text_m:
            ax[2].bar(xs - 0.2, text_m["per_class"], 0.4, color="tab:red", alpha=0.7, label="text-only (seed 42)")
        ax[2].bar(xs + 0.2, mm_pc.mean(0), 0.4, yerr=mm_pc.std(0, ddof=1) if len(mm_pc) > 1 else None, capsize=3,
                  color="tab:blue", alpha=0.8, label=f"text+audio ({len(mm_pc)} seeds, mean ± sd)")
        ax[2].set_xticks(xs, [f"{l}\n(n={n})" for l, n in zip(LABELS, support)], fontsize=8)
        ax[2].set_ylabel("dev F1"); ax[2].set_title("per-class F1"); ax[2].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{tag}_curves.png", dpi=130)
        plt.close()
        print(f"\n已保存 results\\{tag}_epochs.csv / _runs.csv / _steps.csv / _summary.json / _curves.png / "
              f"_dev_probs_s*.npy")
    except Exception as e:
        print(f"(画图跳过：{e})")

    print("\nDay 12 完成 ✅" if all(checks.values()) else "\n有 checklist 没达标 ⚠️ 看上面的输出")


if __name__ == "__main__":
    main()
