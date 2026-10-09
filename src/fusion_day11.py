"""
Day 11 — 多模态 Late Fusion 训练循环：RoBERTa（一起微调）+ wav2vec2 13 层加权和（预提特征）→ 融合 MLP，跑 1 epoch

用法（PowerShell，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python src\\fusion_day11.py --limit 320                          # 冒烟测试（几十秒），输出写 day11_smoke_*，不覆盖正式结果
    python src\\fusion_day11.py                                      # 今日最低目标：1 epoch，方案 A（文本 LN），融合头 lr 1e-3，seed 42
    python src\\fusion_day11.py --scale-fix none text_ln audio_gain  # 对比三种尺度处理（3 次，每次约 40 秒）
    python src\\fusion_day11.py --lr-head 1e-4 3e-4 1e-3             # 计划下午：调融合头学习率（1e-4 ~ 1e-3）
    python src\\fusion_day11.py --freeze-text                        # 计划原文的做法：冻结 RoBERTa，只训融合头（对照）
    python src\\fusion_day11.py --epochs 3                           # 多看几个 epoch 的趋势（Day 12 正式跑 5 epochs）
    python src\\fusion_day11.py --wandb                              # 每次运行一条 wandb run（group=day11）
    （--scale-fix / --lr-head / --seeds 都可以给多个值，脚本跑它们的所有组合，结果写进同一组输出文件）

设定（和文本基线一致，只多了第二组学习率）：
  - AdamW，wd 0.01（1 维参数不加），10% 线性 warmup + 线性衰减（两组参数一起按比例缩放），梯度裁剪 1.0，bf16，batch 16，
    不加 class weights，seed 42
  - 两组学习率：RoBERTa（encoder.*）2e-5；其余新参数（文本 LN / 13 层权重 / 音频 LN / 融合头）--lr-head，默认 1e-3
  - 不冻 RoBERTa（Day 8 交接的"公平比较"：文本基线也是微调过的 RoBERTa）；wav2vec2 是预提特征，本来就冻结
  - 两路尺度（Day 10 发现 text 11.6 vs audio 27.7）：
      none        Day 10 原样
      text_ln     方案 A（默认）：文本 <s> 也过一个 LayerNorm → 两路都≈√768≈27.7
      audio_gain  方案 B：音频 LayerNorm 的 weight 初始化为 11.6/27.7≈0.42 → 音频一开始也≈11.6
只用 dev（1108 条）评估，test 不碰。每次运行：
  - 训练前先评估一次 dev（step 0），之后每 --eval-every 步（默认 125）和每个 epoch 末各评估一次 → 看 val 指标怎么变
  - 每 --log-every 步（默认 50）打印 batch loss / 近 50 步平均 / 两组 lr / 梯度范数（裁剪前）
  - 每次评估还记两路向量的平均 L2 范数，和两路对融合头第一层的"贡献"‖W_m·x_m‖（融合头更听哪一路）
  - 训练完做"打乱测试"：dev 上把音频（或文本）在样本之间随机打乱再评估，wF1 掉得越多 = 模型越依赖这一路
输出（tag = day11，冒烟测试为 day11_smoke）：
  results\\{tag}_steps.csv    每一步的 loss / 梯度范数 / lr
  results\\{tag}_evals.csv    每次 dev 评估：loss / acc / wF1 / mF1 / 各类 F1 / 两路范数和贡献
  results\\{tag}_runs.csv     每次运行一行汇总（最佳 epoch 指标、loss 下降比例、打乱测试、层权重、用时）
  results\\{tag}_curves.png   左：train loss（滑动平均）；右：dev weighted F1 随步数，带文本基线 / 全猜 neutral 参考线
  results\\{tag}_report.json  以上汇总 + 配置
Day 12 起可以直接 import：from fusion_day11 import train_one, predict_mm, SCALE_FIX
Day 12 小改（不影响 Day 11 的结果）：train_one 多一个可选参数 out={}——传了就在训练中记住最佳 epoch 的权重，
  训练完把模型恢复到最佳 epoch 再做打乱测试，并把权重 / dev 概率放进 out；不传时行为和 Day 11 完全一样
"""
import argparse
import copy
import itertools
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from fusion_day10 import (AUDIO_MODEL, LateFusionClassifier, MultimodalDataset, build_table, collate_mm,
                          load_shared_audio, param_groups)
from roberta_day4 import LABELS, MODEL_NAME, OUT_DIR
from text_day6 import metrics, set_seed

K = len(LABELS)
AUDIO_GAIN = round(11.6 / 27.7, 3)  # Day 10 实测：RoBERTa <s> 范数 / 音频 LN 后范数
SCALE_FIX = {
    "none": dict(),
    "text_ln": dict(text_norm=True),
    "audio_gain": dict(audio_ln_gain=AUDIO_GAIN),
}
# Day 10 在 dev 1108 口径下算的参考线（有 day10_check.json 就从文件读）
REF = dict(text=dict(acc=0.6164, wf1=0.6065, mf1=0.4563), neutral=dict(acc=0.4233, wf1=0.2518, mf1=0.0850))


def load_reference():
    p = OUT_DIR / "day10_check.json"
    if p.exists():
        t = json.loads(p.read_text(encoding="utf-8")).get("text_baseline_dev1108")
        if t:
            REF["text"] = dict(acc=t["acc"], wf1=t["wf1"], mf1=t["mf1"])
            REF["neutral"] = t["all_neutral"]
    return REF


# ================================================================ 评估
@torch.no_grad()
def predict_mm(model, loader, device, amp):
    """→ probs (N, 7)、gold (N,)、diag。diag 里每一路 m 有三个数：
      norm_m      这一路向量的平均 L2 范数
      contrib_m   对融合头第一层的平均贡献 ‖W[:, m 的列] · x_m‖（这一路给 256 个隐藏单元带来的输入有多大）
      contribc_m  同上，但 x_m 先减去整个 dev 上的均值（Day 11 加）：所有样本都一样的那部分只相当于一个偏置，
                  不能区分样本；去均值后剩下的才是"随样本变化、可能有用"的贡献
    调用后模型处于 eval 模式。"""
    model.eval()
    W = model.head[1].weight.float()                                     # (256, 1536)
    cols, off = {}, 0
    for m in model.modalities:                                           # 拼接顺序：先文本后音频
        cols[m] = slice(off, off + model.dims[m])
        off += model.dims[m]
    probs, golds = [], []
    feats = {m: [] for m in model.modalities}
    for ids, mask, au, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, parts = model(ids.to(device), mask.to(device), au.to(device))
        probs.append(logits.float().softmax(-1).cpu())
        golds.append(y)
        for m, v in parts.items():
            feats[m].append(v.float())
    diag = {}
    for m, v in feats.items():
        X = torch.cat(v)                                                 # (N, 768)，dev 1108 条约 3MB
        Wm = W[:, cols[m]]
        diag[f"norm_{m}"] = X.norm(dim=1).mean().item()
        diag[f"contrib_{m}"] = (X @ Wm.T).norm(dim=1).mean().item()
        diag[f"contribc_{m}"] = ((X - X.mean(0)) @ Wm.T).norm(dim=1).mean().item()
    return torch.cat(probs).numpy(), torch.cat(golds).numpy(), diag


def shuffled(ds, which, seed=0):
    """ds 的浅拷贝：把一路（"audio" 或 "text"）在样本之间随机打乱，标签和另一路不动。
    这一路和标签的对应关系被破坏 → 如果模型真在用这一路，分数会掉。"""
    perm = np.random.default_rng(seed).permutation(len(ds))
    d = copy.copy(ds)
    if which == "audio":
        d.audio = ds.audio[torch.from_numpy(perm)]
    else:
        d.input_ids = [ds.input_ids[i] for i in perm]
    return d


def fmt_diag(diag):
    mods = [k[5:] for k in diag if k.startswith("norm_")]
    return ("范数 " + " / ".join(f"{m} {diag['norm_' + m]:.1f}" for m in mods)
            + "，贡献 " + " / ".join(f"{m} {diag['contrib_' + m]:.1f}" for m in mods)
            + "，去均值贡献 " + " / ".join(f"{m} {diag['contribc_' + m]:.1f}" for m in mods))


# ================================================================ 训练一次
def run_name(cfg):
    mods = "+".join(cfg["modalities"])
    return (f"{mods}_{cfg['scale_fix']}_lrh{cfg['lr_head']:g}"
            f"{'_frozen' if cfg['freeze_text'] else ''}_e{cfg['epochs']}_s{cfg['seed']}")


def train_one(cfg, train_ds, dev_ds, a, device, amp, pad_id, out=None):
    """cfg：modalities / scale_fix / lr_text / lr_head / freeze_text / epochs / seed。
    返回 (summary dict, steps 列表, evals 列表)。
    out（Day 12 加，可选）：传入一个 dict 时——
      - 每当某个 epoch 的 dev wF1 创新高，就把整个模型的权重拷一份到 CPU（约 500MB 内存）
      - 训练完把模型恢复成最佳 epoch 的权重，打乱测试在最佳 epoch 上做（不传时在最后一个 epoch 上做，和 Day 11 一样）
      - out 里放：state（最佳 epoch 的 state_dict，CPU）、probs（最佳 epoch 的 dev 概率 (N, 7)）、gold（N,）、
        best_epoch、shuffled_probs（{"audio": (N, 7), "text": (N, 7)}，打乱测试的概率）"""
    name = run_name(cfg)
    set_seed(cfg["seed"])                       # 同一个 seed → 融合头初始化相同、batch 顺序相同，配置之间可比
    model = LateFusionClassifier(cfg["modalities"], **SCALE_FIX[cfg["scale_fix"]]).to(device)
    frozen = cfg["freeze_text"] and "text" in model.modalities
    if frozen:
        model.encoder.requires_grad_(False)

    def train_mode():
        model.train()
        if frozen:
            model.encoder.eval()                # 冻结的 RoBERTa 当固定特征提取器：关掉它的 dropout

    col = partial(collate_mm, pad_id=pad_id)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, collate_fn=col, generator=g)
    dev_dl = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=col)

    groups = []
    for gp in param_groups(model, cfg["lr_text"], cfg["lr_head"], a.weight_decay):
        ps = [p for p in gp["params"] if p.requires_grad]  # 冻结时 encoder 的组是空的，丢掉
        if ps:
            groups.append(dict(params=ps, lr=gp["lr"], weight_decay=gp["weight_decay"], name=gp["name"]))
    trainable = [p for gp in groups for p in gp["params"]]
    opt = torch.optim.AdamW(groups)
    total = len(train_dl) * cfg["epochs"]
    warm = int(total * a.warmup)
    sched = get_linear_schedule_with_warmup(opt, warm, total)  # 每组的 lr 都乘同一个系数：warmup 0→1，再线性降到 0
    print(f"  可训练参数 {sum(p.numel() for p in trainable) / 1e6:.2f}M"
          f"（{'无文本' if 'text' not in model.modalities else 'RoBERTa 冻结' if frozen else 'RoBERTa 一起微调'}）；总步数 {total}，warmup {warm} 步；"
          + "；".join(f"{gp['name']} 峰值 lr {gp['initial_lr']:.0e}" for gp in opt.param_groups))
    # （scheduler 建好后 gp["lr"] 已经被乘上 warmup 第 0 步的系数 0，峰值 lr 存在 gp["initial_lr"] 里）

    tag = getattr(a, "tag", "day11")            # Day 12 加：wandb run 名 / group 跟着调用脚本走
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project="emotion-ai", name=f"{tag}-{name}", group=tag, job_type="train",
                        config=dict(cfg, batch=a.batch, warmup=a.warmup, weight_decay=a.weight_decay,
                                    audio_model=a.audio_model, limit=a.limit))

    steps, evals = [], []

    def evaluate(step, ep_float):
        probs, gold, diag = predict_mm(model, dev_dl, device, amp)
        m = metrics(probs, gold)
        evals.append(dict(run=name, step=step, epoch=round(ep_float, 3), dev_loss=round(m["loss"], 5),
                          dev_acc=round(m["acc"], 5), dev_wf1=round(m["wf1"], 5), dev_mf1=round(m["mf1"], 5),
                          **{f"f1_{l}": round(v, 4) for l, v in zip(LABELS, m["per_class"])},
                          **{k: round(v, 3) for k, v in diag.items()}))
        print(f"  [dev] step {step:5d}（epoch {ep_float:.2f}）loss {m['loss']:.4f}  acc {m['acc']:.4f}  "
              f"wF1 {m['wf1']:.4f}  mF1 {m['mf1']:.4f}  | {fmt_diag(diag)}")
        if wb:
            wb.log({"dev/loss": m["loss"], "dev/acc": m["acc"], "dev/wf1": m["wf1"], "dev/mf1": m["mf1"],
                    **{f"diag/{k}": v for k, v in diag.items()}}, step=step)
        train_mode()
        return m, probs, diag

    m0, _, _ = evaluate(0, 0.0)                 # 训练前：val 指标的起点
    best, best_state, step, t0 = None, None, 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        train_mode()
        for i, (ids, mask, au, y) in enumerate(train_dl, 1):
            ids, mask, au, y = ids.to(device), mask.to(device), au.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(ids, mask, au)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()  # 返回裁剪前的总范数
            opt.step()
            sched.step()
            step += 1

            lr = {gp["name"].split("/")[0]: gp["lr"] for gp in opt.param_groups}  # 当前（已按 schedule 缩放的）lr
            steps.append(dict(run=name, step=step, epoch=ep, loss=round(loss.item(), 5), grad_norm=round(gnorm, 4),
                              lr_text=lr.get("text", 0.0), lr_head=lr["head"]))
            if step == 1 or step % a.log_every == 0 or step == total:
                recent = np.mean([s["loss"] for s in steps[-a.log_every:]])
                print(f"  step {step:5d}/{total}  loss {loss.item():.4f}  近 {min(a.log_every, step)} 步平均 {recent:.4f}  "
                      f"lr text {lr.get('text', 0.0):.1e} / head {lr['head']:.1e}  |grad| {gnorm:.2f}")
            if wb:
                wb.log({"train/loss": loss.item(), "train/grad_norm": gnorm, "lr/head": lr["head"],
                        "lr/text": lr.get("text", 0.0)}, step=step)
            if a.eval_every and step % a.eval_every == 0 and i != len(train_dl):
                evaluate(step, ep - 1 + i / len(train_dl))

        m, probs, diag = evaluate(step, float(ep))  # 每个 epoch 末
        pred_n = np.bincount(m["pred"], minlength=K)
        print("  各类 F1（预测条数）：" + "  ".join(f"{l} {f:.3f}({n})" for l, f, n in zip(LABELS, m["per_class"], pred_n)))
        if "audio" in model.modalities:
            w = model.layer_mix.weights().detach().cpu().numpy()
            print("  13 层权重：" + " ".join(f"{x:.3f}" for x in w)
                  + f"  （最大第 {int(w.argmax())} 层，均匀 = {1 / 13:.3f}）")
            if wb:
                wb.log({f"layer_w/{j:02d}": float(x) for j, x in enumerate(w)}, step=step)
        if best is None or m["wf1"] > best["wf1"]:
            best = dict(epoch=ep, probs=probs, diag=diag, **m)
            if out is not None:                 # Day 12：记住最佳 epoch 的权重（拷到 CPU，不占显存）
                best_state = {k_: v.detach().to("cpu", copy=True) for k_, v in model.state_dict().items()}
                print(f"  ★ epoch {ep} 是目前最好的（dev wF1 {m['wf1']:.4f}），已在内存里记下这一刻的权重")
    secs = time.time() - t0

    # Day 12：恢复到最佳 epoch → 下面的打乱测试、层权重、out 都是最佳 epoch 的
    if out is not None:
        if best["epoch"] != cfg["epochs"]:
            model.load_state_dict(best_state)
            print(f"  已把模型恢复到最佳 epoch {best['epoch']} 的权重（最后一个 epoch 是 {cfg['epochs']}）")
        ref_m = best
    else:
        ref_m = m

    # ---------- loss 下降 ----------
    losses = np.array([s["loss"] for s in steps])
    k = max(1, min(50, len(losses) // 4))
    first, last = losses[:k].mean(), losses[-k:].mean()
    k0 = max(1, min(10, k))
    init = losses[:k0].mean()   # "初始值" = 前 10 步平均（warmup 刚开始，lr 很小，≈ ln7）
    drop = 1 - last / init      # Day 11 改：Day 11 第一次跑用的是前 50 步平均，但融合头 lr 1e-3 在前 50 步就学掉了
                                # 一大截（先学会多猜 neutral），拿它当"初始值"会低估下降幅度

    # ---------- 打乱测试（用训练完的模型；传了 out 时是最佳 epoch 的模型）----------
    shuf, shuf_probs = {}, {}
    if len(model.modalities) == 2:
        for which in ("audio", "text"):
            dl = DataLoader(shuffled(dev_ds, which), batch_size=64, shuffle=False, collate_fn=col)
            p, gld, _ = predict_mm(model, dl, device, amp)
            shuf[which] = metrics(p, gld)["wf1"]
            shuf_probs[which] = p
        print(f"  打乱测试（{'最佳 epoch ' + str(best['epoch']) if out is not None else '最后一个 epoch'} 的模型，"
              f"dev wF1 {ref_m['wf1']:.4f}）：打乱音频 → {shuf['audio']:.4f}"
              f"（{shuf['audio'] - ref_m['wf1']:+.4f}），打乱文本 → {shuf['text']:.4f}（{shuf['text'] - ref_m['wf1']:+.4f}）")

    summary = dict(run=name, **{k_: (",".join(v) if k_ == "modalities" else v) for k_, v in cfg.items()},
                   steps=step, secs=round(secs, 1),
                   loss_step1=round(float(losses[0]), 4), loss_init=round(float(init), 4), k0=k0, loss_first_k=round(float(first), 4),
                   loss_last_k=round(float(last), 4), k=k, loss_drop=round(float(drop), 4),
                   dev0_acc=round(m0["acc"], 4), dev0_wf1=round(m0["wf1"], 4),
                   best_epoch=best["epoch"], dev_acc=round(best["acc"], 4), dev_wf1=round(best["wf1"], 4),
                   dev_mf1=round(best["mf1"], 4), dev_loss=round(best["loss"], 4),
                   **{f"f1_{l}": round(v, 4) for l, v in zip(LABELS, best["per_class"])},
                   **{k_: round(v, 3) for k_, v in best["diag"].items()},
                   **({f"shuffled_{w_}_wf1": round(v, 4) for w_, v in shuf.items()}),
                   shuffle_on="best" if out is not None else "last")
    if "audio" in model.modalities:
        summary["layer_weights"] = " ".join(f"{x:.4f}" for x in model.layer_mix.weights().detach().cpu().tolist())
    print(f"  → 用时 {secs / 60:.1f} 分钟；loss 前 {k0} 步平均 {init:.4f}（前 {k} 步 {first:.4f}）→ 最后 {k} 步 {last:.4f}"
          f"（比初始降了 {drop:.0%}）；"
          f"dev acc {m0['acc']:.4f} → {best['acc']:.4f}，wF1 {m0['wf1']:.4f} → {best['wf1']:.4f}（epoch {best['epoch']}）")
    if out is not None:
        out.update(state=best_state if best_state is not None else
                   {k_: v.detach().to("cpu", copy=True) for k_, v in model.state_dict().items()},
                   probs=best["probs"], gold=np.asarray(dev_ds.labels), best_epoch=best["epoch"],
                   shuffled_probs=shuf_probs)
    if wb:
        wb.summary.update({k_: v for k_, v in summary.items() if isinstance(v, (int, float))})
        wb.finish()
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary, steps, evals


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", default=["text", "audio"], choices=["text", "audio"])
    ap.add_argument("--scale-fix", nargs="+", default=["text_ln"], choices=list(SCALE_FIX))
    ap.add_argument("--lr-head", nargs="+", type=float, default=[1e-3])
    ap.add_argument("--lr-text", type=float, default=2e-5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--freeze-text", action="store_true", help="冻结 RoBERTa，只训融合部分（计划原文的做法）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=125, help="每 N 步评估一次 dev（0 = 只在 epoch 末）")
    ap.add_argument("--audio-model", default=AUDIO_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试：train 随机 N 条、dev 前 N 条")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.no_amp
    print(f"device={device}  bf16={amp}  HF_HOME={os.environ['HF_HOME']}  audio={a.audio_model}")
    OUT_DIR.mkdir(exist_ok=True)
    lm = OUT_DIR / "label_map.json"
    if lm.exists():
        saved = json.loads(lm.read_text(encoding="utf-8"))["labels"]
        assert saved == LABELS, f"label 顺序和 label_map.json 不一致：{saved} vs {LABELS}"
    ref = load_reference()
    tag = "day11_smoke" if a.limit else "day11"

    # ---------- 数据（只用 train / dev，test 不碰）----------
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
    del train_emb, dev_emb                      # Dataset 里已经按 df 顺序拷了一份
    print(f"  train {len(train_ds)} / dev {len(dev_ds)}；每 epoch {math.ceil(len(train_ds) / a.batch)} 步")
    print(f"  参考线（dev 1108 口径）：文本基线 acc {ref['text']['acc']:.4f} / wF1 {ref['text']['wf1']:.4f} / "
          f"mF1 {ref['text']['mf1']:.4f}；全猜 neutral wF1 {ref['neutral']['wf1']:.4f}")

    # ---------- 所有组合 ----------
    cfgs = [dict(modalities=tuple(m for m in ("text", "audio") if m in a.modalities), scale_fix=sf,
                 lr_text=a.lr_text, lr_head=lrh, freeze_text=a.freeze_text, epochs=a.epochs, seed=s)
            for sf, lrh, s in itertools.product(a.scale_fix, a.lr_head, a.seeds)]
    print(f"\n共 {len(cfgs)} 次运行（scale_fix {a.scale_fix} × lr_head {a.lr_head} × seeds {a.seeds}），"
          f"每次 {a.epochs} epoch")

    runs, all_steps, all_evals = [], [], []
    t_all = time.time()
    for j, cfg in enumerate(cfgs, 1):
        print(f"\n========== {j}/{len(cfgs)}  {run_name(cfg)} ==========")
        s, st, ev = train_one(cfg, train_ds, dev_ds, a, device, amp, tok.pad_token_id)
        runs.append(s)
        all_steps += st
        all_evals += ev
    print(f"\n全部用时 {(time.time() - t_all) / 60:.1f} 分钟")
    gpu_peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None
    if gpu_peak:
        print(f"显存峰值 {gpu_peak} GB")

    runs_df = pd.DataFrame(runs)
    pd.DataFrame(all_steps).to_csv(OUT_DIR / f"{tag}_steps.csv", index=False)
    pd.DataFrame(all_evals).to_csv(OUT_DIR / f"{tag}_evals.csv", index=False)
    runs_df.to_csv(OUT_DIR / f"{tag}_runs.csv", index=False)

    # ---------- 汇总 ----------
    print("\n========== 汇总（dev；Δ 是和文本基线 wF1 {:.4f} 比）==========".format(ref["text"]["wf1"]))
    has_shuf = "shuffled_audio_wf1" in runs_df
    print(f"  {'run':40s} {'loss降':>6} {'acc0→':>6} {'acc':>6} {'wF1':>6} {'Δ':>7} {'mF1':>6}"
          + (f" {'乱音频':>7} {'乱文本':>7}" if has_shuf else ""))
    for r in runs_df.itertuples():
        line = (f"  {r.run:40s} {r.loss_drop:>6.0%} {r.dev0_acc:>6.3f} {r.dev_acc:>6.3f} {r.dev_wf1:>6.4f} "
                f"{r.dev_wf1 - ref['text']['wf1']:>+7.4f} {r.dev_mf1:>6.4f}")
        if has_shuf:
            line += f" {r.shuffled_audio_wf1 - r.dev_wf1:>+7.4f} {r.shuffled_text_wf1 - r.dev_wf1:>+7.4f}"
        print(line)
    if has_shuf:
        print("  （乱音频 / 乱文本 = 打乱那一路后 wF1 的变化；越负 = 越依赖那一路）")
    if len(a.seeds) > 1:
        print("\n  多个 seed 合并（均值 ± 标准差）：")
        for key, g in runs_df.groupby(["scale_fix", "lr_head"], sort=False):
            print(f"    {key[0]:10s} lr_head {key[1]:g}: wF1 {g.dev_wf1.mean():.4f} ± {g.dev_wf1.std(ddof=1):.4f}，"
                  f"mF1 {g.dev_mf1.mean():.4f} ± {g.dev_mf1.std(ddof=1):.4f}（n={len(g)}）")

    # ---------- Day 11 checklist（第一个运行）----------
    r0 = runs[0]
    checks = {
        f"{a.epochs} epoch 训练完成": r0["steps"] == math.ceil(len(train_ds) / a.batch) * a.epochs,
        f"loss 明显下降（最后 {r0['k']} 步比前 {r0['k0']} 步降了 {r0['loss_drop']:.0%}，要求 ≥ 33%）": r0["loss_drop"] >= 1 / 3,
        f"val accuracy 有改善（{r0['dev0_acc']:.4f} → {r0['dev_acc']:.4f}）": r0["dev_acc"] > r0["dev0_acc"],
    }
    print(f"\nDay 11 checklist（{r0['run']}）：")
    for k_, v in checks.items():
        print(f"  {'✅' if v else '⚠️'} {k_}")

    report = dict(args=vars(a), reference=ref, runs=runs, checks={k_: bool(v) for k_, v in checks.items()},
                  gpu_peak_GB=gpu_peak, text_model=MODEL_NAME, saved_at=time.strftime("%Y-%m-%d %H:%M"))
    (OUT_DIR / f"{tag}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        st, ev = pd.DataFrame(all_steps), pd.DataFrame(all_evals)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for name, g in st.groupby("run", sort=False):
            win = max(1, min(50, len(g) // 10))
            ax[0].plot(g.step, g.loss.rolling(win, min_periods=1).mean(), label=name, lw=1.2)
        ax[0].axhline(math.log(K), ls="--", c="gray", lw=0.8, label="ln 7")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("train loss (moving avg 50)"); ax[0].set_title("train loss")
        for name, g in ev.groupby("run", sort=False):
            ax[1].plot(g.step, g.dev_wf1, marker="o", ms=4, label=name)
        ax[1].axhline(ref["text"]["wf1"], ls="--", c="tab:red", lw=1, label=f"text-only baseline {ref['text']['wf1']:.3f}")
        ax[1].axhline(ref["neutral"]["wf1"], ls=":", c="gray", lw=1, label=f"all-neutral {ref['neutral']['wf1']:.3f}")
        ax[1].set_xlabel("step"); ax[1].set_ylabel("dev weighted F1"); ax[1].set_title(f"dev weighted F1 ({len(dev_ds)} utts)")
        for x in ax:
            x.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{tag}_curves.png", dpi=130)
        plt.close()
        print(f"\n已保存 {tag}_steps.csv / _evals.csv / _runs.csv / _report.json / _curves.png（results\\）")
    except Exception as e:
        print(f"(画图跳过：{e})")

    print("\nDay 11 完成 ✅" if all(checks.values()) else "\n有 checklist 没达标 ⚠️ 看上面的输出")


if __name__ == "__main__":
    main()
