"""
Day 14 — test set 最终评估：用事先定好的 seed 42 权重，在 MELD test（2610 条）上只评一次

用法（PowerShell，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python src\\test_day14.py --limit 200                 # 冒烟测试：test / dev 各取前 200 条，输出写 test_smoke_*（1 分钟内）
    python src\\test_day14.py                             # 今日最低目标：seed 42 的模型在 test 上评估（约 1–2 分钟）
    python src\\test_day14.py --retrain-seeds 43 44       # 可选：用和 Day 12/13 完全相同的配置重训 seed 43 / 44
                                                          #   （text+audio / text-only / audio-only，约 12 分钟），训完直接在 test 上
                                                          #   推理 → test 也有 3 seed 均值 ± 标准差。test 概率存成 .npy，再跑不重训

规则（为什么这样做才算\"诚实的 test 数字\"）：
  - 所有选择都已经在 dev 上做完：配置（Day 11）、最佳 epoch（每个 seed 按 dev wF1）、保存哪个 seed（事先定 42）。
    test 只拿来报数，看完不再改任何东西
  - 重训 seed 43 / 44 不是在 test 上挑模型：最佳 epoch 仍按 dev 选；脚本还会核对重训出来的 dev wF1 和
    Day 12/13 的 {tag}_runs.csv 是否一致（训练可复现，Day 6 / 11 验证过）
  - temperature scaling 的 T 在 dev 上拟合，test 只用来看校准有没有变好（给 Day 16 Gradio 的概率条用）
参与评估的模型（都是 seed 42、按 dev wF1 选的最佳 epoch）：
  text + audio   results\\fusion_best.pt             （Day 12，epoch 5）
  text-only      results\\fusion_day13_text.pt       （Day 13，同一个 LateFusionClassifier，只有文本那一路）
  audio-only     results\\fusion_day13_audio_e20.pt  （Day 13，20 epochs 版）
  可选：fusion_day12_frozen.pt / fusion_day13_text_frozen.pt（冻结 RoBERTa 的 2×2，文件在就评）
  参照：text_best.pt（Day 6 TextClassifier）
  每个模型先在 dev 上重新推理一次，和 .json 里记录的 dev wF1 核对 → 确认权重加载对了，再去 test
做的事：
  1. 汇总表：dev wF1（训练时记的）vs test acc / wF1 / mF1，以及去掉 shared_audio（42 条）/ 只去掉跨 split 泄漏的数字
  2. 每个模型的 sklearn classification_report（各类 P / R / F1）+ 混淆矩阵 → results\\test_results.txt
  3. 配对 bootstrap：text+audio vs text-only 等，同一批 test 句子重采样（有多 seed 时用 3 seed 平均版）
  4. 各类 F1：test 上的 Δ（text+audio − text-only）和 dev 上的 Δ 并排 → dev 上 anger 的提升在 test 上复现了吗
  5. 混淆矩阵差值（text+audio − text-only）+ 最难区分的情绪对（计划下午：\"找出模型最难区分的情绪对\"）
  6. 校准：dev 上拟合 temperature T，看 test 上 NLL / ECE 前后 → results\\fusion_best_temperature.json（Day 16 用）
  7. 计划可选：3 类简化（positive / negative / neutral），直接把 7 类预测合并，不重新训练
输出（冒烟测试前缀为 test_smoke）：
  results\\test_results.txt          各模型 classification report + 混淆矩阵 + 汇总（英文，进仓库）
  results\\test_summary.json         以上全部数字
  results\\test_probs_{模型}_s{seed}.npy   test 概率 (2610, 7)，行顺序 = build_table(\"test\")（被 .gitignore 排除）
  results\\test_errors.csv           text+audio（seed 42）在 test 上的错例，按置信度从高到低
  results\\test_confusion.png        text+audio / text-only 按行归一化 + 差值
  results\\test_per_class.png        各类 F1：test 柱状图 + dev 的点
  results\\fusion_best_temperature.json   在 dev 上拟合的 T
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from ablation_day13 import paired_boot_mean, scores
from fusion_day10 import AUDIO_MODEL, LateFusionClassifier, MultimodalDataset, build_table, collate_mm, load_shared_audio
from fusion_day11 import predict_mm, train_one
from roberta_day4 import LABELS, MODEL_NAME, OUT_DIR, TextClassifier
from text_day6 import predict

try:  # 和 text_day6 一样兼容本机旧文件名
    from epoch1_day5 import TextDataset, collate
except ImportError:
    from text_day5 import TextDataset, collate

K = len(LABELS)
# key, 表里的名字, 权重文件名（不带后缀）, Day 12/13 的结果 tag（核对 dev、重训其他 seed 时对照 runs.csv）, 是否可选
SPEC = [
    ("mm", "text + audio", "fusion_best", "day12", False),
    ("text", "text-only", "fusion_day13_text", "day13_text", False),
    ("audio", "audio-only", "fusion_day13_audio_e20", "day13_audio_e20", False),
    ("frozen", "text + audio (frozen RoBERTa)", "fusion_day12_frozen", "day12_frozen", True),
    ("tfrozen", "text-only (frozen RoBERTa)", "fusion_day13_text_frozen", "day13_text_frozen", True),
]
NAME = {k: n for k, n, *_ in SPEC}
NAME["d6"] = "text-only · TextClassifier (Day 6)"
# 计划 Appendix A 的 3 类简化：positive = joy / surprise，negative = anger / disgust / fear / sadness
G3 = ["negative", "neutral", "positive"]
GROUP3 = dict(anger="negative", disgust="negative", fear="negative", sadness="negative",
              neutral="neutral", joy="positive", surprise="positive")
MAP3 = np.array([G3.index(GROUP3[l]) for l in LABELS])          # 7 类 id → 3 类 id


# ================================================================ 加载 / 推理
def load_meta(name):
    """→ (meta dict, .pt 路径) 或 None（两个文件缺一个）。"""
    js, pt = OUT_DIR / f"{name}.json", OUT_DIR / f"{name}.pt"
    if not (js.exists() and pt.exists()):
        return None
    return json.loads(js.read_text(encoding="utf-8")), pt


def build_fusion(meta, state, device):
    """按 .json 里存的 config / model_kwargs 建模型（不手写参数），再装权重。state 是文件路径或 state_dict。"""
    model = LateFusionClassifier(tuple(meta["config"]["modalities"]), **meta["model_kwargs"])
    if not isinstance(state, dict):
        state = torch.load(state, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def infer_mm(model, ds, device, amp, pad_id):
    """→ probs (N, 7)，行顺序和 ds 一样（shuffle=False）。"""
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=partial(collate_mm, pad_id=pad_id))
    probs, _, _ = predict_mm(model, dl, device, amp)
    return probs


def infer_text(model, df, tok, device, amp, max_length):
    """Day 6 的 TextClassifier 只吃文本：用 TextDataset + collate（和 Day 6 训练时一样）。"""
    dl = DataLoader(TextDataset(df, tok, max_length), batch_size=64, shuffle=False,
                    collate_fn=partial(collate, pad_id=tok.pad_token_id))
    probs, _ = predict(model, dl, device, amp)
    return probs


def free(device):
    if device == "cuda":
        torch.cuda.empty_cache()


# ================================================================ 校准（temperature scaling）
def nll(probs, gold):
    return float(-np.log(probs[np.arange(len(gold)), gold] + 1e-12).mean())


def apply_t(probs, T):
    """softmax(log p / T)。log p 和 logits 只差每行一个常数，softmax 不受影响，所以不用存 logits。
    T > 1 把概率\"压平\"（不那么自信），T < 1 让它更尖；argmax 不变 → acc / F1 完全不变。"""
    z = np.log(probs + 1e-12) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(probs, gold):
    """在 dev 上找让 NLL 最小的 T（0.25 ~ 8 之间对数均匀取 701 个点，简单、不会陷进局部最小）。"""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 701))
    losses = [nll(apply_t(probs, t), gold) for t in grid]
    return float(grid[int(np.argmin(losses))])


def ece(probs, gold, n_bins=15):
    """Expected Calibration Error：按最大概率分 15 个区间，每个区间 |准确率 − 平均置信度| 按条数加权求和。
    0 = 完美校准（说 80% 把握的预测正好对 80%）。"""
    conf, right = probs.max(1), probs.argmax(1) == gold
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(right[m].mean() - conf[m].mean())
    return float(e)


# ================================================================ 小工具
def ms(x, d=4):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def mean_cm(gold, probs_list):
    return np.mean([confusion_matrix(gold, p.argmax(1), labels=range(K)) for p in probs_list], axis=0)


def scores3(gold, pred):
    """7 类 → 3 类后再算 acc / wF1 / mF1。"""
    g, p = MAP3[gold], MAP3[pred]
    lab = list(range(3))
    return dict(acc=float((g == p).mean()),
                wf1=float(f1_score(g, p, labels=lab, average="weighted", zero_division=0)),
                mf1=float(f1_score(g, p, labels=lab, average="macro", zero_division=0)))


def cm_text(cm, fmt="d", signed=False):
    short = [l[:5] for l in LABELS]
    lines = ["gold\\pred " + " ".join(f"{s:>6}" for s in short)]
    lines += [f"{LABELS[i]:9s} " + " ".join(format(v, (">+6" if signed else ">6") + fmt) for v in row) for i, row in enumerate(cm)]
    return "\n".join(lines)


# ================================================================ 重训其他 seed（可选）
def retrain(key, seed, meta, tag, train_ds, dev_ds, test_ds, a, device, amp, pad_id):
    """用 meta 里的配置、换 seed 重训一次（fusion_day11.train_one，和 Day 12/13 同一个函数）。
    → (test 概率, 记录)；记录里有这次的 dev wF1 / 最佳 epoch 和 Day 12/13 runs.csv 里同一个 seed 的数字。"""
    cfg = dict(meta["config"], modalities=tuple(meta["config"]["modalities"]), seed=seed)
    ta = argparse.Namespace(batch=meta.get("batch", 16), warmup=meta.get("warmup", 0.1), weight_decay=0.01,
                            log_every=a.log_every, eval_every=0, wandb=False, audio_model=a.audio_model,
                            limit=a.limit, tag="day14_retrain")
    out = {}
    s, _, _ = train_one(cfg, train_ds, dev_ds, ta, device, amp, pad_id, out=out)
    model = build_fusion(meta, out["state"], device)
    probs = infer_mm(model, test_ds, device, amp, pad_id)
    del model, out
    free(device)
    rec = dict(dev_wf1=s["dev_wf1"], best_epoch=s["best_epoch"])
    rp = OUT_DIR / f"{tag}_runs.csv"
    if rp.exists():
        r = pd.read_csv(rp)
        r = r[r.seed == seed]
        if len(r):
            rec.update(logged_dev_wf1=float(r.dev_wf1.iloc[0]), logged_best_epoch=int(r.best_epoch.iloc[0]))
    return probs, rec


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain-seeds", nargs="*", type=int, default=[], help="可选：再重训这些 seed（如 43 44）")
    ap.add_argument("--retrain-models", nargs="+", default=["mm", "text", "audio"], choices=["mm", "text", "audio"])
    ap.add_argument("--force-retrain", action="store_true", help="已有 test_probs_*_s{seed}.npy 也重训")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--n-errors", type=int, default=10, help="打印多少条置信度最高的错例")
    ap.add_argument("--audio-model", default=AUDIO_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试：test / dev 前 N 条、train 随机 N 条")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    prefix = "test_smoke" if a.limit else "test"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.no_amp
    t_all = time.time()
    print(f"device={device}  bf16={amp}  HF_HOME={os.environ['HF_HOME']}  输出前缀 {prefix}_*")

    # ---------- 数据 ----------
    print("\n========== 数据 ==========")
    dups = load_shared_audio()
    test_df, test_emb, info = build_table("test", a.audio_model, dups)
    dev_df, dev_emb, _ = build_table("dev", a.audio_model, dups)
    if a.limit:
        test_df, dev_df = test_df.head(a.limit).reset_index(drop=True), dev_df.head(a.limit).reset_index(drop=True)
        print(f"  冒烟测试：test {len(test_df)} / dev {len(dev_df)} 条（核对 dev 时和 .json 对不上是正常的）")
    tok = RobertaTokenizer.from_pretrained(MODEL_NAME)
    pad_id = tok.pad_token_id
    test_ds = MultimodalDataset(test_df, test_emb, tok, a.max_length)
    dev_ds = MultimodalDataset(dev_df, dev_emb, tok, a.max_length)
    del test_emb, dev_emb
    gold = np.asarray(test_ds.labels)
    dev_gold = np.asarray(dev_ds.labels)
    shared = test_df["shared_audio"].to_numpy(dtype=bool)
    cross = test_df["dup_cross_split"].to_numpy(dtype=bool)
    support = np.bincount(gold, minlength=K)
    print(f"  test {len(gold)} 条：" + " ".join(f"{l} {n}" for l, n in zip(LABELS, support)))
    print(f"  shared_audio {shared.sum()} 条，其中和 train 共用音频（跨 split 泄漏）{cross.sum()} 条")
    neutral_wf1 = scores(gold, np.full(len(gold), LABELS.index("neutral")))["wf1"]
    print(f"  全猜 neutral：test wF1 {neutral_wf1:.4f}")

    # ---------- 1. 加载 seed 42 权重：先在 dev 上核对，再在 test 上推理 ----------
    print("\n========== 1. seed 42 权重：dev 核对 → test 推理 ==========")
    P, DEV, META = {}, {}, {}           # P[key][seed] = test 概率；DEV[key] = seed 42 的 dev 概率
    dev_check = {}
    for key, name, fname, tag, optional in SPEC:
        lm = load_meta(fname)
        if lm is None:
            if optional:
                print(f"  （可选）{fname}.pt / .json 不在，跳过 {name}")
                continue
            raise SystemExit(f"  ❌ 找不到 results\\{fname}.pt / .json（{name}），先按 Day 12/13 的命令训练")
        meta, pt = lm
        assert int(meta["config"]["seed"]) == 42, f"{fname}.json 的 seed 是 {meta['config']['seed']}，不是事先定好的 42"
        META[key] = dict(meta, tag=tag, fname=fname)
        t0 = time.time()
        model = build_fusion(meta, pt, device)
        dprobs = infer_mm(model, dev_ds, device, amp, pad_id)
        tprobs = infer_mm(model, test_ds, device, amp, pad_id)
        del model
        free(device)
        DEV[key], P[key] = dprobs, {int(meta["config"]["seed"]): tprobs}
        dw = scores(dev_gold, dprobs.argmax(1))["wf1"]
        ok = abs(dw - meta["dev_wf1"]) < 1e-3
        same = None                       # 和 Day 12/13 存的 dev 概率逐条比 argmax
        pp = OUT_DIR / f"{tag}_dev_probs_s{meta['config']['seed']}.npy"
        if pp.exists() and not a.limit:
            old = np.load(pp)
            if old.shape == dprobs.shape:
                same = float((old.argmax(1) == dprobs.argmax(1)).mean())
        dev_check[key] = dict(dev_wf1_now=round(dw, 4), dev_wf1_json=meta["dev_wf1"], ok=bool(ok), same_pred=same)
        tw = scores(gold, tprobs.argmax(1))["wf1"]
        print(f"  {name:30s} ← {fname}.pt（seed {meta['config']['seed']}，epoch {meta['best_epoch']}）  "
              f"dev wF1 {dw:.4f} vs json {meta['dev_wf1']:.4f} {'✅' if ok else '⚠️'}"
              + (f"，和 {pp.name} 预测一致 {same:.1%}" if same is not None else "")
              + f"  → test wF1 {tw:.4f}（{time.time() - t0:.0f}s）")

    # Day 6 TextClassifier（参照）
    p6 = OUT_DIR / "text_best.pt"
    if p6.exists():
        t0 = time.time()
        tm = TextClassifier().to(device)
        tm.load_state_dict(torch.load(p6, map_location="cpu"))
        DEV["d6"] = infer_text(tm, dev_df, tok, device, amp, a.max_length)
        P["d6"] = {42: infer_text(tm, test_df, tok, device, amp, a.max_length)}
        del tm
        free(device)
        dw = scores(dev_gold, DEV["d6"].argmax(1))["wf1"]
        dev_check["d6"] = dict(dev_wf1_now=round(dw, 4), dev_wf1_json=0.6065, ok=bool(abs(dw - 0.6065) < 1e-3))
        print(f"  {NAME['d6']:30s} ← text_best.pt  dev wF1 {dw:.4f}（应为 0.6065，1108 口径）"
              f"{'✅' if dev_check['d6']['ok'] else '⚠️'}  → test wF1 {scores(gold, P['d6'][42].argmax(1))['wf1']:.4f}"
              f"（{time.time() - t0:.0f}s）")
    else:
        print("  （text_best.pt 不在，跳过 Day 6 参照行）")
    if not a.limit and not all(v["ok"] for v in dev_check.values()):
        print("  ⚠️ 有模型的 dev wF1 和记录对不上 → 权重可能被覆盖过，先查清楚再看 test 数字")

    # ---------- 2. 可选：重训其他 seed ----------
    retrain_log = {}
    log_p = OUT_DIR / f"{prefix}_retrain_log.json"
    if log_p.exists():
        retrain_log = json.loads(log_p.read_text(encoding="utf-8"))
    if a.retrain_seeds:
        print(f"\n========== 2. 重训 seed {a.retrain_seeds}（{', '.join(NAME[k] for k in a.retrain_models)}）==========")
        train_ds = None
        for key in a.retrain_models:
            for seed in a.retrain_seeds:
                f = OUT_DIR / f"{prefix}_probs_{key}_s{seed}.npy"
                lk = f"{key}_s{seed}"
                if f.exists() and lk in retrain_log and not a.force_retrain:
                    P[key][seed] = np.load(f)
                    print(f"  {NAME[key]} seed {seed}：已有 {f.name}，直接读（--force-retrain 重训）")
                    continue
                if train_ds is None:              # 真要重训时才读 train（200MB）
                    train_df, train_emb, _ = build_table("train", a.audio_model, dups, verbose=False)
                    if a.limit:
                        train_df = train_df.sample(n=min(a.limit, len(train_df)), random_state=42).reset_index(drop=True)
                    train_ds = MultimodalDataset(train_df, train_emb, tok, a.max_length)
                    del train_emb
                print(f"\n  ---- {NAME[key]}  seed {seed}（配置来自 {META[key]['fname']}.json）----")
                t0 = time.time()
                probs, rec = retrain(key, seed, META[key], META[key]["tag"], train_ds, dev_ds, test_ds, a, device,
                                     amp, pad_id)
                rec["minutes"] = round((time.time() - t0) / 60, 1)
                P[key][seed] = probs
                np.save(f, probs)
                retrain_log[lk] = rec
                log_p.write_text(json.dumps(retrain_log, indent=2), encoding="utf-8")   # 每次都存，中断了下次接着跑
        print("\n  重训核对（dev wF1：这次 vs Day 12/13 runs.csv；一致说明训练可复现，test 均值没有混进别的东西）：")
        for lk, rec in retrain_log.items():
            lg = rec.get("logged_dev_wf1")
            flag = "" if lg is None else ("✅" if abs(rec["dev_wf1"] - lg) < 2e-3 else "⚠️ 不一致")
            print(f"    {lk:10s} dev wF1 {rec['dev_wf1']:.4f}（epoch {rec['best_epoch']}）"
                  + (f" vs 记录 {lg:.4f}（epoch {rec.get('logged_best_epoch')}）{flag}" if lg is not None else ""))
    for key in list(P):
        for f in sorted(OUT_DIR.glob(f"{prefix}_probs_{key}_s*.npy")):     # 以前重训过的 seed 也读进来
            s = int(f.stem.rsplit("_s", 1)[1])
            if s not in P[key] and f"{key}_s{s}" in retrain_log:
                P[key][s] = np.load(f)
    for key, d in P.items():                                               # seed 42 的 test 概率也存一份
        for s, p in d.items():
            np.save(OUT_DIR / f"{prefix}_probs_{key}_s{s}.npy", p)
    multi = {k: sorted(d) for k, d in P.items() if len(d) > 1}
    if multi:
        print(f"  有多个 seed 的模型：" + "，".join(f"{NAME[k]} {v}" for k, v in multi.items()))

    # ---------- 3. 汇总表 ----------
    print("\n========== 3. 汇总（test；seed 42 = 事先定好的那一个模型）==========")
    rows = []
    for key, d in P.items():
        m42 = scores(gold, d[42].argmax(1))
        ns = scores(gold[~shared], d[42][~shared].argmax(1))
        nc = scores(gold[~cross], d[42][~cross].argmax(1))
        allw = [scores(gold, p.argmax(1))["wf1"] for p in d.values()]
        allm = [scores(gold, p.argmax(1))["mf1"] for p in d.values()]
        alla = [scores(gold, p.argmax(1))["acc"] for p in d.values()]
        dev42 = META[key]["dev_wf1"] if key in META else dev_check.get("d6", {}).get("dev_wf1_now")
        rows.append(dict(key=key, model=NAME[key], seeds=" ".join(map(str, sorted(d))),
                         dev_wf1_s42=dev42, test_acc_s42=m42["acc"], test_wf1_s42=m42["wf1"], test_mf1_s42=m42["mf1"],
                         gap_s42=m42["wf1"] - dev42 if dev42 is not None else np.nan,
                         test_wf1_no_shared=ns["wf1"], test_wf1_no_cross=nc["wf1"],
                         test_acc_mean=np.mean(alla), test_wf1_mean=np.mean(allw), test_mf1_mean=np.mean(allm),
                         test_wf1_sd=np.std(allw, ddof=1) if len(allw) > 1 else np.nan,
                         test_mf1_sd=np.std(allm, ddof=1) if len(allm) > 1 else np.nan,
                         test_wf1_all=" ".join(f"{x:.4f}" for x in allw)))
    tab = pd.DataFrame(rows)
    print(f"  {'模型':34s} {'dev wF1':>7} {'test acc':>8} {'test wF1':>8} {'test mF1':>8} {'test−dev':>8}"
          f" {'去shared':>8} {'去泄漏':>7}")
    for r in tab.itertuples():
        print(f"  {r.model:34s} {r.dev_wf1_s42:>7.4f} {r.test_acc_s42:>8.4f} {r.test_wf1_s42:>8.4f} "
              f"{r.test_mf1_s42:>8.4f} {r.gap_s42:>+8.4f} {r.test_wf1_no_shared:>8.4f} {r.test_wf1_no_cross:>7.4f}")
    print(f"  {'全猜 neutral':34s} {'':>7} {'':>8} {neutral_wf1:>8.4f}")
    print(f"  （去shared = 去掉 {shared.sum()} 条 shared_audio，n={(~shared).sum()}；"
          f"去泄漏 = 只去掉和 train 共用音频的 {cross.sum()} 条）")
    if multi:
        print("  多 seed（每个 seed 的最佳 epoch 都按 dev 选）：")
        for r in tab[tab.key.isin(multi)].itertuples():
            print(f"    {r.model:32s} seeds {r.seeds}：test acc {r.test_acc_mean:.4f}，wF1 {r.test_wf1_mean:.4f} ± "
                  f"{r.test_wf1_sd:.4f}，mF1 {r.test_mf1_mean:.4f} ± {r.test_mf1_sd:.4f}（各 seed：{r.test_wf1_all}）")
    tab.to_csv(OUT_DIR / f"{prefix}_summary_table.csv", index=False)

    # ---------- 4. classification report ----------
    print("\n========== 4. classification report（seed 42）==========")
    reports, cms = {}, {}
    for key, d in P.items():
        pred = d[42].argmax(1)
        reports[key] = classification_report(gold, pred, labels=list(range(K)), target_names=LABELS, digits=3,
                                             zero_division=0)
        cms[key] = confusion_matrix(gold, pred, labels=range(K))
    for key in ("mm", "text"):
        print(f"  ---- {NAME[key]} ----")
        print(reports[key])
        print(cm_text(cms[key]))
        print()

    # ---------- 5. 配对 bootstrap ----------
    print(f"========== 5. 配对 bootstrap（同一批 {len(gold)} 条 test 句子，{a.bootstrap} 次）==========")
    comps = [("mm", "text", "text+audio vs text-only (what audio adds)"),
             ("mm", "d6", "text+audio vs Day 6 TextClassifier"),
             ("mm", "audio", "text+audio vs audio-only (what text adds)"),
             ("text", "audio", "text-only vs audio-only"),
             ("frozen", "tfrozen", "frozen RoBERTa: text+audio vs text-only")]
    boots = {}
    for ka, kb, desc in comps:
        if ka not in P or kb not in P or not a.bootstrap:
            continue
        b42 = paired_boot_mean(gold, [P[ka][42]], [P[kb][42]], n=a.bootstrap)
        res = dict(desc=desc, s42=b42)
        line = f"    seed 42：Δ wF1 {b42['diff']:+.4f}，95% [{b42['lo']:+.4f}, {b42['hi']:+.4f}]，Δ ≤ 0 的比例 {b42['p_le0']:.3f}"
        common = sorted(set(P[ka]) & set(P[kb]))
        if len(common) > 1:
            bm = paired_boot_mean(gold, [P[ka][s] for s in common], [P[kb][s] for s in common], n=a.bootstrap)
            res["mean"] = dict(bm, seeds=common)
            line += (f"\n    {len(common)} seed 平均：Δ wF1 {bm['diff']:+.4f}，95% [{bm['lo']:+.4f}, {bm['hi']:+.4f}]，"
                     f"Δ ≤ 0 的比例 {bm['p_le0']:.3f}")
        main_b = res.get("mean", b42)
        res["verdict"] = "CI excludes 0" if not (main_b["lo"] <= 0 <= main_b["hi"]) else "CI includes 0"
        boots[f"{ka}_vs_{kb}"] = res
        print(f"  {desc}\n{line}\n    → {'区间不含 0：test 上站得住' if res['verdict'] == 'CI excludes 0' else '区间含 0：噪声范围内'}")
    dev_mt = None
    sp = OUT_DIR / "day13_summary.json"
    d13 = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    if d13.get("bootstrap", {}).get("mm_vs_text"):
        dev_mt = d13["bootstrap"]["mm_vs_text"]
        print(f"  对照 dev（Day 13，3 seed 平均）：Δ {dev_mt['diff']:+.4f}，95% [{dev_mt['lo']:+.4f}, {dev_mt['hi']:+.4f}]")

    # ---------- 6. 各类 F1：test vs dev ----------
    print("\n========== 6. 各类 F1（test；有多 seed 时取平均）+ dev 上的 Δ 对照 ==========")
    pcs = {k: np.array([scores(gold, p.argmax(1))["per_class"] for p in d.values()]) for k, d in P.items()}
    dev_pc = {}
    pcp = OUT_DIR / "day13_per_class.csv"
    if pcp.exists():
        dev_pc = pd.read_csv(pcp).set_index("label")["delta_mm_text"].to_dict()
    keys3 = [k for k in ("text", "audio", "mm") if k in P]
    pc_rows = []
    print(f"  {'类别':9s} {'n':>4} " + " ".join(f"{NAME[k][:12]:>12}" for k in keys3) + f" {'Δ test':>8} {'Δ dev':>7}")
    for i, l in enumerate(LABELS):
        row = dict(label=l, support=int(support[i]), **{f"{k}": round(float(pcs[k][:, i].mean()), 4) for k in keys3})
        row["delta_test"] = round(row["mm"] - row["text"], 4)
        row["delta_dev"] = dev_pc.get(l, np.nan)
        pc_rows.append(row)
        print(f"  {l:9s} {support[i]:>4d} " + " ".join(f"{row[k]:>12.3f}" for k in keys3)
              + f" {row['delta_test']:>+8.3f} {row['delta_dev']:>+7.3f}")
    pc = pd.DataFrame(pc_rows)
    pc.to_csv(OUT_DIR / f"{prefix}_per_class.csv", index=False)
    seeds_note = f"{len(P['mm'])} seed 平均" if len(P["mm"]) > 1 else "seed 42"
    print(f"  （Δ = text+audio − text-only，test 是 {seeds_note}；Δ dev 是 Day 13 的 3 seed 平均）")
    anger = pc.set_index("label").loc["anger"]
    print(f"  dev 上提升最大的 anger：dev Δ {anger.delta_dev:+.3f} → test Δ {anger.delta_test:+.3f}"
          f"  {'→ 方向复现' if anger.delta_test > 0 else '→ test 上没复现'}")

    # ---------- 7. 混淆矩阵差值 + 最难区分的情绪对 ----------
    print("\n========== 7. 混淆矩阵 ==========")
    common = sorted(set(P["mm"]) & set(P["text"]))
    cm_mm, cm_t = mean_cm(gold, [P["mm"][s] for s in common]), mean_cm(gold, [P["text"][s] for s in common])
    diff = cm_mm - cm_t
    n_c = f"{len(common)} seed 平均条数" if len(common) > 1 else "seed 42 条数"
    print(f"  text+audio − text-only（{n_c}；行 = 真实，列 = 预测）：")
    print("  " + cm_text(diff, ".1f", signed=True).replace("\n", "\n  "))
    ia, isad = LABELS.index("anger"), LABELS.index("sadness")
    dev_diff = np.array(d13["confusion_diff_mm_minus_text"]) if "confusion_diff_mm_minus_text" in d13 else None
    print(f"  anger 行：对角 {diff[ia, ia]:+.1f}，anger→sadness {diff[ia, isad]:+.1f}"
          + (f"（dev 上是 对角 {dev_diff[ia, ia]:+.1f}、anger→sadness {dev_diff[ia, isad]:+.1f}）" if dev_diff is not None else ""))
    cmn = cms["mm"] / cms["mm"].sum(1, keepdims=True).clip(min=1)
    pairs = sorted(((cms["mm"][i, j], i, j) for i in range(K) for j in range(K) if i != j), reverse=True)[:8]
    print("  text+audio（seed 42）最常见的错误（真实 → 预测）：")
    for n, i, j in pairs:
        print(f"    {LABELS[i]:9s} → {LABELS[j]:9s} {n:4d} 条（占该真实类 {cmn[i, j]:.0%}）")
    sym = []
    for i in range(K):
        for j in range(i + 1, K):
            sym.append(((cms["mm"][i, j] + cms["mm"][j, i]) / (support[i] + support[j]), i, j))
    sym.sort(reverse=True)
    print("  最难区分的情绪对（两个方向的错误条数 ÷ 两类总条数）：" + "，".join(
        f"{LABELS[i]}↔{LABELS[j]} {r:.1%}" for r, i, j in sym[:5]))
    ineu = LABELS.index("neutral")
    non_neu_to_neu = int(cms["mm"][:, ineu].sum() - cms["mm"][ineu, ineu])   # neutral 那一列，去掉对角线
    wrong = int((P["mm"][42].argmax(1) != gold).sum())
    print(f"  错例共 {wrong} 条，其中被判成 neutral 的 {non_neu_to_neu} 条（{non_neu_to_neu / max(wrong, 1):.0%}）"
          f"→ 和 Day 6 一样，最主要的错误方向仍是塌向 neutral")

    # ---------- 8. 校准 ----------
    print("\n========== 8. 校准（T 在 dev 上拟合，test 只用来检验）==========")
    calib = {}
    for key in [k for k in ("mm", "text") if k in DEV]:
        T = fit_temperature(DEV[key], dev_gold)
        pt42 = P[key][42]
        calib[key] = dict(T=round(T, 3), dev_nll_before=nll(DEV[key], dev_gold),
                          dev_nll_after=nll(apply_t(DEV[key], T), dev_gold),
                          test_nll_before=nll(pt42, gold), test_nll_after=nll(apply_t(pt42, T), gold),
                          test_ece_before=ece(pt42, gold), test_ece_after=ece(apply_t(pt42, T), gold),
                          test_mean_conf_before=float(pt42.max(1).mean()),
                          test_mean_conf_after=float(apply_t(pt42, T).max(1).mean()), test_acc=float((pt42.argmax(1) == gold).mean()))
        c = calib[key]
        print(f"  {NAME[key]:14s} T = {T:.2f}；test NLL {c['test_nll_before']:.3f} → {c['test_nll_after']:.3f}，"
              f"ECE {c['test_ece_before']:.3f} → {c['test_ece_after']:.3f}，"
              f"平均置信度 {c['test_mean_conf_before']:.3f} → {c['test_mean_conf_after']:.3f}（acc {c['test_acc']:.3f}，不变）")
    if "mm" in calib and not a.limit:
        (OUT_DIR / "fusion_best_temperature.json").write_text(json.dumps(dict(
            temperature=calib["mm"]["T"], fitted_on="dev (1108 utterances, fusion_best.pt seed 42)",
            usage="probs = softmax(logits / T)", **{k: round(v, 4) for k, v in calib["mm"].items() if k != "T"},
            saved_at=time.strftime("%Y-%m-%d %H:%M")), indent=2), encoding="utf-8")
        print("  已存 results\\fusion_best_temperature.json（Day 16 Gradio 的概率条用 softmax(logits / T)）")

    # ---------- 9. 3 类简化（计划可选）----------
    print("\n========== 9. 3 类简化（positive = joy/surprise，negative = anger/disgust/fear/sadness；直接合并 7 类预测）==========")
    three = {}
    for key in [k for k in ("text", "audio", "mm") if k in P]:
        three[key] = scores3(gold, P[key][42].argmax(1))
        print(f"  {NAME[key]:14s} acc {three[key]['acc']:.4f}  wF1 {three[key]['wf1']:.4f}  mF1 {three[key]['mf1']:.4f}")
    print("  （7 类 wF1 已远高于 Appendix A 的 0.40 阈值，主结果仍报 7 类；这一节只作参考）")

    # ---------- 10. 错例 ----------
    pm = P["mm"][42]
    pred_mm = pm.argmax(1)
    err = test_df[["Dialogue_ID", "Utterance_ID", "Speaker", "text", "duration_s", "shared_audio"]].copy()
    err["gold"] = [LABELS[i] for i in gold]
    err["mm_pred"] = [LABELS[i] for i in pred_mm]
    err["mm_conf"] = pm.max(1).round(3)
    err["mm_p_gold"] = pm[np.arange(len(gold)), gold].round(3)
    err["text_pred"] = [LABELS[i] for i in P["text"][42].argmax(1)]
    err["audio_pred"] = [LABELS[i] for i in P["audio"][42].argmax(1)]
    err = err[pred_mm != gold].sort_values("mm_conf", ascending=False)
    err.to_csv(OUT_DIR / f"{prefix}_errors.csv", index=False, encoding="utf-8-sig")
    print(f"\n========== 10. text+audio 错例 {len(err)} 条（已存 {prefix}_errors.csv），置信度最高的 {a.n_errors} 条 ==========")
    for r in err.head(a.n_errors).itertuples():
        print(f"  [{r.gold} → {r.mm_pred} {r.mm_conf:.2f}] dia{r.Dialogue_ID}_utt{r.Utterance_ID}  "
              f"text-only→{r.text_pred}  audio-only→{r.audio_pred}  {str(r.text)[:70]}")
    words = test_df["text"].str.split().str.len().to_numpy()
    bins = [(1, 2, "1–2 词"), (3, 5, "3–5 词"), (6, 10, "6–10 词"), (11, 10 ** 6, "11+ 词")]
    print("  按句长的 test acc：" + "  ".join(
        f"{lab} {((pred_mm == gold)[(words >= lo) & (words <= hi)]).mean():.3f}（{((words >= lo) & (words <= hi)).sum()} 条）"
        for lo, hi, lab in bins if ((words >= lo) & (words <= hi)).any()))

    # ---------- test_results.txt ----------
    main_rows = [r for r in rows if r["key"] in ("mm", "text", "audio", "d6")] + \
                [r for r in rows if r["key"] in ("frozen", "tfrozen")]
    L = [f"MELD test set — final evaluation (Day 14, {time.strftime('%Y-%m-%d %H:%M')})",
         f"n = {len(gold)} test utterances (all have audio). Checkpoints are the seed-42 models whose best epoch was "
         "chosen on dev before the test set was touched; nothing was tuned on test.",
         f"Always-neutral baseline: test weighted F1 {neutral_wf1:.4f}", "",
         "== Summary (seed 42) ==",
         f"{'model':36s} {'dev wF1':>8} {'test acc':>9} {'test wF1':>9} {'test mF1':>9} {'wF1 w/o shared audio':>21}"]
    L += [f"{r['model']:36s} {r['dev_wf1_s42']:>8.4f} {r['test_acc_s42']:>9.4f} {r['test_wf1_s42']:>9.4f} "
          f"{r['test_mf1_s42']:>9.4f} {r['test_wf1_no_shared']:>21.4f}" for r in main_rows]
    if multi:
        L += ["", "== Multiple seeds (mean ± sd; best epoch per seed chosen on dev) =="]
        L += [f"{r['model']:36s} seeds {r['seeds']}: test wF1 {r['test_wf1_mean']:.4f} ± {r['test_wf1_sd']:.4f}, "
              f"mF1 {r['test_mf1_mean']:.4f} ± {r['test_mf1_sd']:.4f}" for r in rows if r["key"] in multi]
    if boots:
        L += ["", f"== Paired bootstrap over test utterances ({a.bootstrap} resamples) =="]
        for v in boots.values():
            b = v.get("mean", v["s42"])
            L.append(f"{v['desc']}: Δ wF1 {b['diff']:+.4f}, 95% CI [{b['lo']:+.4f}, {b['hi']:+.4f}]"
                     + (f" ({len(v['mean']['seeds'])}-seed mean)" if "mean" in v else " (seed 42)"))
    L += ["", "== Per-class F1 (test) ==",
          f"{'class':9s} {'n':>5} " + " ".join(f"{NAME[k][:12]:>12}" for k in keys3) + f" {'Δ test':>8} {'Δ dev':>7}"]
    L += [f"{r['label']:9s} {r['support']:>5d} " + " ".join(f"{r[k]:>12.3f}" for k in keys3)
          + f" {r['delta_test']:>+8.3f} {r['delta_dev']:>+7.3f}" for r in pc_rows]
    if "mm" in calib:
        c = calib["mm"]
        L += ["", f"== Calibration (text + audio, temperature fitted on dev: T = {c['T']:.2f}) ==",
              f"test NLL {c['test_nll_before']:.3f} -> {c['test_nll_after']:.3f}; "
              f"ECE {c['test_ece_before']:.3f} -> {c['test_ece_after']:.3f}"]
    for key in [k for k in ("mm", "text", "audio", "d6", "frozen", "tfrozen") if k in reports]:
        L += ["", f"== {NAME[key]} — classification report (seed 42) ==", reports[key],
              "Confusion matrix (rows = gold, columns = predicted):", cm_text(cms[key])]
    (OUT_DIR / f"{prefix}_results.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def heat(ax, mat, title, fmt, cmap, vmin, vmax):
            ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=7)
            ax.set_xticks(range(K), LABELS, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(K), LABELS, fontsize=8)
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
        for ax, key in zip(axs[:2], ("mm", "text")):
            heat(ax, cms[key] / cms[key].sum(1, keepdims=True).clip(min=1),
                 f"{NAME[key]} — test, row-normalised\nweighted F1 {scores(gold, P[key][42].argmax(1))['wf1']:.3f} (seed 42)",
                 ".2f", "Blues", 0, 1)
        vmax = max(1.0, float(np.abs(diff).max()))
        heat(axs[2], diff, f"text+audio − text-only (utterance counts, {len(common)}-seed mean)", "+.1f", "RdBu", -vmax, vmax)
        plt.tight_layout(); plt.savefig(OUT_DIR / f"{prefix}_confusion.png", dpi=130); plt.close()

        fig, ax = plt.subplots(figsize=(11, 4.4))
        xs = np.arange(K)
        cols = dict(text="tab:red", audio="tab:green", mm="tab:blue")
        wbar = 0.8 / len(keys3)
        for j, k in enumerate(keys3):
            x = xs + (j - (len(keys3) - 1) / 2) * wbar
            ax.bar(x, pcs[k].mean(0), wbar, yerr=pcs[k].std(0, ddof=1) if len(pcs[k]) > 1 else None, capsize=2,
                   color=cols[k], alpha=0.85, label=f"{NAME[k]} — test")
            dv = META[k].get("per_class_f1") if k in META else None
            if dv:
                ax.scatter(x, [dv[l] for l in LABELS], marker="_", s=180, c="black", zorder=3,
                           label="dev (seed 42)" if j == 0 else None)
        ax.set_xticks(xs, [f"{l}\n(n={n})" for l, n in zip(LABELS, support)], fontsize=8)
        n_s = len(P["mm"])
        ax.set_ylabel("F1")
        ax.set_title(f"per-class F1 — MELD test (bars, {f'{n_s}-seed mean ± sd' if n_s > 1 else 'seed 42'}) "
                     f"vs dev (black ticks, seed 42)")
        ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(OUT_DIR / f"{prefix}_per_class.png", dpi=130); plt.close()
    except Exception as e:
        print(f"(画图跳过：{e})")

    # ---------- summary.json ----------
    summary = dict(n_test=len(gold), n_shared=int(shared.sum()), n_cross=int(cross.sum()), always_neutral_wf1=neutral_wf1,
                   dev_check=dev_check, table=rows, bootstrap=boots, dev_bootstrap_mm_vs_text=dev_mt, per_class=pc_rows,
                   confusion_mm=cms["mm"].tolist(), confusion_diff_mm_minus_text=diff.round(2).tolist(),
                   hardest_pairs=[dict(pair=f"{LABELS[i]}-{LABELS[j]}", rate=round(float(r), 4)) for r, i, j in sym[:5]],
                   calibration=calib, three_class=three, retrain=retrain_log, seeds={k: sorted(d) for k, d in P.items()},
                   minutes=round((time.time() - t_all) / 60, 1), saved_at=time.strftime("%Y-%m-%d %H:%M"))
    (OUT_DIR / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                                    encoding="utf-8")

    # ---------- 结论 + checklist ----------
    print("\n========== 结论（PS / README 用的数字）==========")
    r_mm, r_t = tab.set_index("key").loc["mm"], tab.set_index("key").loc["text"]
    mt = boots.get("mm_vs_text")
    if mt:
        b = mt.get("mean", mt["s42"])
        print(f"  text+audio test wF1 {r_mm.test_wf1_s42:.4f}（seed 42）"
              + (f"，{len(P['mm'])} seed {r_mm.test_wf1_mean:.4f} ± {r_mm.test_wf1_sd:.4f}" if "mm" in multi else "")
              + f"；text-only {r_t.test_wf1_s42:.4f}"
              + (f"，{len(P['text'])} seed {r_t.test_wf1_mean:.4f} ± {r_t.test_wf1_sd:.4f}" if "text" in multi else ""))
        print(f"  加音频：Δ wF1 {b['diff'] * 100:+.1f} 个百分点，95% 区间 [{b['lo'] * 100:+.1f}, {b['hi'] * 100:+.1f}] → "
              + ("区间不含 0，可以写\"提升在 test 上站得住\"" if mt["verdict"] == "CI excludes 0"
                 else "区间含 0，按 Day 13 的口径写\"小幅、不显著\""))
        print("  PS 模板里的 \"YY% improvement\" 建议写成绝对百分点（不是相对百分比），并附上区间")
    print(f"  dev → test 的差距（seed 42 wF1）：" + "，".join(f"{r.model} {r.gap_s42:+.4f}" for r in tab.itertuples()))
    checks = {"test set evaluation 完成": all(k in P for k in ("mm", "text", "audio")),
              "classification_report 打印出来": "mm" in reports and "text" in reports,
              f"结果文件保存（results\\{prefix}_results.txt）": (OUT_DIR / f"{prefix}_results.txt").exists()}
    print(f"\n已保存 results\\{prefix}_results.txt / _summary.json / _summary_table.csv / _per_class.csv / _errors.csv / "
          f"_confusion.png / _per_class.png / _probs_*.npy（共 {(time.time() - t_all) / 60:.1f} 分钟）")
    print("\nDay 14 checklist：")
    for k_, v in checks.items():
        print(f"  {'✅' if v else '⚠️'} {k_}")
    print("\nDay 14 完成 ✅" if all(checks.values()) else "\n有 checklist 没达标 ⚠️")


if __name__ == "__main__":
    main()
