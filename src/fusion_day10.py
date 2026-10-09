"""
Day 10 — Late Fusion：MultimodalDataset（文本 + 音频 13 层 + 标签）+ LateFusionClassifier 的 forward / backward 检查

用法（PowerShell，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python src\\fusion_day10.py                       # 今日全部检查（4090 上约 1–2 分钟）
    python src\\fusion_day10.py --overfit-steps 0     # 跳过"单 batch 过拟合"检查
    python src\\fusion_day10.py --wandb               # 可选：wandb 初始化 multimodal 实验（先 wandb login）

按 Day 9 交接的设计：
  - 文本和音频按 (Dialogue_ID, Utterance_ID) merge，不按行号；缺 wav 的 train dia125_utt3、dev dia110_utt7 自然丢掉
    → 多模态 train 9988 / dev 1108 / test 2610
  - 读 results\\day9_audio_duplicates.csv，给每条样本加 shared_audio 标记（只标记，不删；错误分析时单独看）
  - 音频 13 层用可学习的加权和（softmax 权重，初始化为均匀 = mean13）→ LayerNorm → 768 维
  - 融合：concat(RoBERTa <s> 768, audio 768) = 1536 → Dropout → Linear(1536, 256) → ReLU → Dropout → Linear(256, 7)
  - RoBERTa 一起微调（lr 2e-5），音频加权和 + LayerNorm + 融合头用更大的 lr（1e-3）——两组参数两个学习率
  - 同一个类用 modalities=("text",) / ("audio",) 就是 Day 13 的 text-only / audio-only，三路消融共用一套代码
  - Day 11 新增：text_norm=True 让文本 <s> 也过一个 LayerNorm（修两路尺度不一致）；audio_ln_gain 是另一种修法
    （默认值 text_norm=False / audio_ln_gain=1.0 = Day 10 原样，所以本脚本的检查结果不变）
做的检查：
  1. 三个 split 的 text ⋈ audio 对齐：条数、丢掉的 key、标签一致、shared_audio 数量
  2. MultimodalDataset[0] 返回 (input_ids, audio (13, 768), label)；DataLoader 完整遍历 train 一遍
  3. LateFusionClassifier forward：(batch, 7) logits；均匀初始化时加权和 == mean13；三种 modalities 都不报维度错
  4. loss + backward：梯度能传到 RoBERTa、层权重、融合头（并且融合头第一层的文本列 / 音频列都有梯度）
  5. 单 batch 过拟合：同一个 batch 训 30 步，loss 应接近 0（训练代码没接错的最低保证，Day 11 的前提）
  6. 文本基线在同一口径（dev 1108 条，去掉 dia110_utt7）下重算 → Day 12 和多模态比的就是这个数
输出：
  results\\day10_check.json   以上所有数字
Day 11 起直接 import：
  from fusion_day10 import build_table, MultimodalDataset, collate_mm, LateFusionClassifier, param_groups
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer

from audio_day9 import load_audio_feats
from roberta_day4 import LABELS, MODEL_NAME, OUT_DIR, TextClassifier, load_split
from text_day6 import metrics, predict, set_seed

try:  # 和 text_day6 一样兼容本机旧文件名
    from epoch1_day5 import TextDataset, collate
except ImportError:
    from text_day5 import TextDataset, collate

AUDIO_MODEL = "facebook/wav2vec2-base"
SPLITS = ["train", "dev", "test"]
KEY = ["Dialogue_ID", "Utterance_ID"]
K = len(LABELS)
EXPECTED = dict(train=9988, dev=1108, test=2610)  # Day 9 提取的条数 = CSV − 缺 wav


# ================================================================ 数据
def load_shared_audio():
    """Day 9 的重复音频表 → 每个 (split, key) 一行：dup_group、组是否跨 split、组内标签是否冲突。"""
    p = OUT_DIR / "day9_audio_duplicates.csv"
    if not p.exists():
        print(f"  ⚠️ 找不到 {p.name}（先跑 python src\\check_dups_day9.py），shared_audio 全部记为 False")
        return None
    d = pd.read_csv(p, encoding="utf-8-sig")
    g = d.groupby("h")
    d["dup_cross_split"] = g.split.transform("nunique") > 1
    d["dup_label_conflict"] = g.emotion.transform("nunique") > 1
    return d.rename(columns={"h": "dup_group"})[
        ["split", *KEY, "dup_group", "dup_cross_split", "dup_label_conflict"]]


def build_table(split, audio_model=AUDIO_MODEL, dups=None, verbose=True):
    """文本 CSV ⋈ 音频特征（按 key inner join）。
    返回 (df, emb, info)：df 一行一条样本，含 text / label / audio_idx（emb 的行号）/ row_csv（原 CSV 行号）/ shared_audio 等；
    emb 是 (N_audio, 13, 768) float16，用 df.audio_idx 取；info 是条数 / 丢掉的 key / shared_audio 数量等统计。"""
    text = load_split(split)
    text["row_csv"] = np.arange(len(text))  # 原 CSV 行号：对齐 Day 6 存的 day6_dev_probs.npy (1109, 7) 要用
    emb, keys = load_audio_feats(split, audio_model)
    keys = keys.rename(columns={"label": "audio_label"})
    keys["audio_idx"] = np.arange(len(keys))
    # validate="one_to_one"：两边 key 都必须唯一，否则直接报错（防止一条文本配到两段音频）
    df = text.merge(keys, on=KEY, how="inner", validate="one_to_one")
    lost = text.merge(keys[KEY], on=KEY, how="left", indicator=True)
    lost = lost[lost["_merge"] == "left_only"]
    n_label_mismatch = int((df.label != df.audio_label).sum())
    assert n_label_mismatch == 0, f"[{split}] 有 {n_label_mismatch} 条文本标签和音频文件里的标签不一致"

    if dups is not None:
        df = df.merge(dups[dups.split == split].drop(columns="split"), on=KEY, how="left")
    else:
        df["dup_group"] = np.nan
        df["dup_cross_split"] = False
        df["dup_label_conflict"] = False
    df["shared_audio"] = df["dup_group"].notna()
    df["dup_cross_split"] = df["dup_cross_split"].fillna(False).astype(bool)
    df["dup_label_conflict"] = df["dup_label_conflict"].fillna(False).astype(bool)
    df = df.drop(columns="audio_label").reset_index(drop=True)

    info = dict(n_csv=len(text), n_audio=len(keys), n=len(df),
                dropped=[f"dia{d}_utt{u}" for d, u in zip(lost.Dialogue_ID, lost.Utterance_ID)],
                shared_audio=int(df.shared_audio.sum()), shared_cross_split=int(df.dup_cross_split.sum()),
                shared_label_conflict=int(df.dup_label_conflict.sum()), truncated=int(df.truncated.sum()))
    if verbose:
        ok = len(df) == EXPECTED.get(split, len(df))
        print(f"  [{split:5s}] CSV {info['n_csv']} 条，音频 {info['n_audio']} 条 → 对齐后 {info['n']} 条 "
              f"{'✅' if ok else '⚠️ 和 Day 9 记录的 ' + str(EXPECTED[split]) + ' 不一致'}；"
              f"丢掉 {', '.join(info['dropped']) or '无'}；标签一致 ✅；"
              f"shared_audio {info['shared_audio']} 条（跨 split {info['shared_cross_split']}，"
              f"所在组标签冲突 {info['shared_label_conflict']}）；截断到 20s 的 {info['truncated']} 条")
    return df, emb, info


class MultimodalDataset(Dataset):
    """__getitem__ → (input_ids: list[int], audio: (13, 768) float16 tensor, label: int)。
    文本只 tokenize 不 pad（collate 时动态 padding，和文本基线完全一样）；
    音频特征整份放内存（train 约 200MB float16），按 df.audio_idx 重新排成和 df 同一顺序。"""

    def __init__(self, df, audio_emb, tokenizer, max_length=128):
        enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.input_ids = enc["input_ids"]
        self.audio = torch.from_numpy(np.ascontiguousarray(audio_emb[df["audio_idx"].to_numpy()]))
        self.labels = df["label"].astype(int).tolist()
        self.shared_audio = df["shared_audio"].to_numpy()  # 不进 batch，评估时按样本顺序（shuffle=False）对回去

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.audio[i], self.labels[i]


def collate_mm(batch, pad_id):
    """→ input_ids (B, L) / attention_mask (B, L) / audio (B, 13, 768) float16 / labels (B,)"""
    L = max(len(ids) for ids, _, _ in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (ids, _, _) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
    audio = torch.stack([a for _, a, _ in batch])
    labels = torch.tensor([y for _, _, y in batch], dtype=torch.long)
    return input_ids, mask, audio, labels


# ================================================================ 模型
class LayerWeightedSum(nn.Module):
    """SUPERB 的做法：13 个层各一个可学习的标量，softmax 后对各层加权求和。
    初始 logits 全 0 → 权重全是 1/13 → 一开始就等于 Day 9 探针里的 mean13。"""

    def __init__(self, n_layers=13):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def weights(self):
        return self.logits.softmax(0)

    def forward(self, x):                                   # x: (B, 13, 768)
        return torch.einsum("l,bld->bd", self.weights(), x.float())


class LateFusionClassifier(nn.Module):
    """modalities=("text", "audio") 为多模态；("text",) / ("audio",) 为 Day 13 的单模态消融。
    forward 返回 (logits (B, 7), parts)，parts 是 {"text": (B, 768), "audio": (B, 768)}，错误分析/画图用。
    两路尺度（Day 10 实测：RoBERTa <s> 的 L2 范数约 11.6，音频过 LayerNorm 后约 27.7 ≈ √768）：
      text_norm=True      文本 <s> 也过一个 LayerNorm → 两路都≈√768（Day 11 方案 A）
      audio_ln_gain=0.42  音频 LayerNorm 的 weight 初始化成 0.42 → 音频一开始也≈11.6（方案 B）"""

    def __init__(self, modalities=("text", "audio"), n_classes=K, n_layers=13, audio_dim=768,
                 hidden=256, dropout=0.1, model_name=MODEL_NAME, text_norm=False, audio_ln_gain=1.0):
        super().__init__()
        assert modalities and set(modalities) <= {"text", "audio"}, modalities
        self.modalities = tuple(m for m in ("text", "audio") if m in modalities)  # 固定顺序：先文本后音频
        dims = {}
        self.text_norm = None
        if "text" in self.modalities:
            self.encoder = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
            dims["text"] = self.encoder.config.hidden_size
            if text_norm:
                self.text_norm = nn.LayerNorm(dims["text"])
        if "audio" in self.modalities:
            self.layer_mix = LayerWeightedSum(n_layers)
            # LayerNorm 把每条音频向量的 768 维拉到均值 0、方差 1（范数≈√768≈27.7），
            # 注意这比 RoBERTa <s> 的范数（≈11.6）大，两路要对齐尺度见 text_norm / audio_ln_gain
            self.audio_norm = nn.LayerNorm(audio_dim)
            nn.init.constant_(self.audio_norm.weight, audio_ln_gain)
            dims["audio"] = audio_dim
        self.dims = dims
        in_dim = sum(dims.values())
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, input_ids=None, attention_mask=None, audio=None):
        parts = {}
        if "text" in self.modalities:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            t = h[:, 0, :]                                               # <s> 向量 (B, 768)
            parts["text"] = self.text_norm(t) if self.text_norm is not None else t
        if "audio" in self.modalities:
            parts["audio"] = self.audio_norm(self.layer_mix(audio))     # (B, 768)
        fused = torch.cat([parts[m] for m in self.modalities], dim=-1)  # (B, 1536)
        return self.head(fused), parts


def param_groups(model, lr_text=2e-5, lr_head=1e-3, weight_decay=0.01):
    """两个学习率 × 是否 weight decay。1 维参数（bias、LayerNorm、层权重 logits）不加 decay，
    对 RoBERTa 来说和 Day 5 的 ("bias", "LayerNorm.weight") 规则等价。
    只有 encoder.*（RoBERTa）用 lr_text；text_norm / layer_mix / audio_norm / head 都是新参数，用 lr_head。"""
    groups = {}
    for n, p in model.named_parameters():
        part = "text" if n.startswith("encoder.") else "head"
        decay = p.ndim >= 2
        key = (part, decay)
        if key not in groups:
            groups[key] = dict(params=[], names=[], lr=lr_text if part == "text" else lr_head,
                               weight_decay=weight_decay if decay else 0.0, name=f"{part}/{'decay' if decay else 'no_decay'}")
        groups[key]["params"].append(p)
        groups[key]["names"].append(n)
    return list(groups.values())


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-model", default=AUDIO_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lr-text", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--overfit-steps", type=int, default=30, help="单 batch 过拟合检查的步数，0 = 跳过")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.no_amp
    print(f"device={device}  bf16={amp}  HF_HOME={os.environ['HF_HOME']}  audio={a.audio_model}")
    OUT_DIR.mkdir(exist_ok=True)
    rep = dict(audio_model=a.audio_model, text_model=MODEL_NAME, batch=a.batch, seed=a.seed,
               lr_text=a.lr_text, lr_head=a.lr_head)
    checks = {}

    # ---------- 1. 对齐 ----------
    print("\n========== 1. 文本 ⋈ 音频（按 Dialogue_ID, Utterance_ID）==========")
    dups = load_shared_audio()
    tables, embs, rep["align"] = {}, {}, {}
    for s in SPLITS:
        tables[s], embs[s], rep["align"][s] = build_table(s, a.audio_model, dups)
    checks["align"] = all(len(tables[s]) == EXPECTED[s] for s in SPLITS)
    shared = pd.concat([t.assign(split=s) for s, t in tables.items()])
    shared = shared[shared.shared_audio]
    print(f"  shared_audio 合计 {len(shared)} 条（Day 9 记录 85 条：train 41 / dev 2 / test 42）"
          f"{' ✅' if len(shared) == 85 else ' ⚠️'}；只做标记，不删")

    # ---------- 2. Dataset + DataLoader ----------
    print("\n========== 2. MultimodalDataset + DataLoader ==========")
    tok = RobertaTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    ds = {s: MultimodalDataset(tables[s], embs[s], tok, a.max_length) for s in SPLITS}
    print(f"  建 Dataset 用时 {time.time() - t0:.1f}s：" + "，".join(f"{s} {len(d)}" for s, d in ds.items()))
    ids0, au0, y0 = ds["train"][0]
    r0 = tables["train"].iloc[0]
    print(f"  train_ds[0] → input_ids 长度 {len(ids0)}，audio {tuple(au0.shape)} {au0.dtype}，"
          f"label {y0} ({LABELS[y0]})   dia{r0.Dialogue_ID}_utt{r0.Utterance_ID}: {r0.text!r}")
    z0, _ = load_audio_feats("train", a.audio_model)
    same0 = bool(torch.equal(au0, torch.from_numpy(z0[r0.audio_idx])))
    del z0
    print(f"  audio 和 .npz 里同一个 key 的那一行逐位相同：{'✅' if same0 else '⚠️'}")
    checks["getitem"] = au0.shape == (13, 768) and isinstance(y0, int) and same0

    col = partial(collate_mm, pad_id=tok.pad_token_id)
    g = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(ds["train"], batch_size=a.batch, shuffle=True, collate_fn=col, generator=g)
    t0, nb = time.time(), 0
    for ids, mask, au, y in train_dl:
        nb += 1
    print(f"  train 遍历完毕：{nb} 个 batch（{time.time() - t0:.1f}s）；最后一个 batch：input_ids {tuple(ids.shape)}，"
          f"audio {tuple(au.shape)} {au.dtype}，labels {tuple(y.shape)}")
    checks["dataloader"] = nb == math.ceil(len(ds["train"]) / a.batch)
    rep["dataset"] = dict(n={s: len(d) for s, d in ds.items()}, train_batches=nb, first_len=len(ids0))

    # 固定一个 batch 给后面的 forward / backward / 过拟合检查用
    fixed = next(iter(DataLoader(ds["train"], batch_size=a.batch, shuffle=True, collate_fn=col,
                                 generator=torch.Generator().manual_seed(0))))
    ids, mask, au, y = (t.to(device) for t in fixed)

    # ---------- 3. 模型 + forward ----------
    print("\n========== 3. LateFusionClassifier forward ==========")
    set_seed(a.seed)
    model = LateFusionClassifier().to(device)
    print(f"  参数量：RoBERTa {n_params(model.encoder) / 1e6:.1f}M，层权重 {n_params(model.layer_mix)}，"
          f"音频 LayerNorm {n_params(model.audio_norm)}，融合头 {n_params(model.head):,}"
          f"（1536×256+256 + 256×7+7 = {1536 * 256 + 256 + 256 * 7 + 7:,}）")
    w0 = model.layer_mix.weights().detach()
    mix_ok = torch.allclose(model.layer_mix(au), au.float().mean(1), atol=1e-4)
    print(f"  初始层权重全是 1/13={1 / 13:.4f}：{'✅' if torch.allclose(w0, torch.full_like(w0, 1 / 13)) else '⚠️'}；"
          f"加权和 == 13 层直接平均（mean13）：{'✅' if mix_ok else '⚠️'}")

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, parts = model(ids, mask, au)
    print(f"  input_ids {tuple(ids.shape)} + audio {tuple(au.shape)} → text <s> {tuple(parts['text'].shape)}，"
          f"audio {tuple(parts['audio'].shape)} → concat (B, 1536) → logits {tuple(logits.shape)}"
          f"  {'✅' if logits.shape == (len(y), K) else '⚠️'}")
    tn, an = parts["text"].float().norm(dim=1).mean().item(), parts["audio"].float().norm(dim=1).mean().item()
    raw = au.float().mean(1).norm(dim=1).mean().item()
    print(f"  向量长度（L2 范数，batch 平均）：text {tn:.1f}，audio LayerNorm 前 {raw:.1f} → 后 {an:.1f}"
          f"（≈√768={math.sqrt(768):.1f}）——拼接前两路尺度要在同一个量级，否则融合头会偏向大的那一路")
    p0 = logits.float().softmax(-1)[0].tolist()
    print("  未训练时第 1 条的概率（应接近 1/7≈0.143）：", {LABELS[i]: round(v, 3) for i, v in enumerate(p0)})
    checks["forward"] = logits.shape == (len(y), K) and mix_ok
    rep["forward"] = dict(logits=list(logits.shape), text_norm=round(tn, 2), audio_norm_before=round(raw, 2),
                          audio_norm_after=round(an, 2), head_params=n_params(model.head))

    # 三种 modalities 都能 forward（Day 13 消融用同一个类）
    shapes = {}
    for mods in [("audio",), ("text",)]:
        m = LateFusionClassifier(mods).to(device).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lg, _ = m(ids, mask, au)
        shapes["+".join(mods)] = (list(lg.shape), m.head[1].in_features)
        del m
    shapes["text+audio"] = (list(logits.shape), model.head[1].in_features)
    for k, (s, d) in shapes.items():
        print(f"  modalities={k:10s} 融合头输入 {d:4d} 维 → logits {tuple(s)}")
    checks["modalities"] = all(s == [len(y), K] for s, _ in shapes.values())
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------- 4. loss + backward ----------
    print("\n========== 4. loss + backward ==========")
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, _ = model(ids, mask, au)
    loss = F.cross_entropy(logits.float(), y)
    loss.backward()
    W1 = model.head[1].weight.grad                                      # (256, 1536)
    gr = dict(
        roberta_word_emb=model.encoder.embeddings.word_embeddings.weight.grad.norm().item(),
        layer_weights=model.layer_mix.logits.grad.norm().item(),
        audio_layernorm=model.audio_norm.weight.grad.norm().item(),
        head_in_text_cols=W1[:, :768].norm().item(),
        head_in_audio_cols=W1[:, 768:].norm().item(),
        head_out=model.head[4].weight.grad.norm().item(),
    )
    print(f"  loss {loss.item():.4f}（ln7 = {math.log(7):.4f}，接近即正常）")
    for k, v in gr.items():
        print(f"    梯度范数 {k:20s} {v:.4g}")
    checks["backward"] = all(v > 0 and math.isfinite(v) for v in gr.values())
    print(f"  全部 > 0 且有限：{'✅ 梯度传到了 RoBERTa、层权重和融合头的文本列 / 音频列' if checks['backward'] else '⚠️'}")
    rep["backward"] = dict(loss=round(loss.item(), 4), grad_norms={k: round(v, 6) for k, v in gr.items()})

    # ---------- 5. 单 batch 过拟合 ----------
    if a.overfit_steps:
        print(f"\n========== 5. 单 batch 过拟合（同一个 batch 训 {a.overfit_steps} 步）==========")
        groups = param_groups(model, a.lr_text, a.lr_head)
        for gp in groups:
            print(f"  参数组 {gp['name']:16s} lr {gp['lr']:.0e}  wd {gp['weight_decay']}  "
                  f"{sum(p.numel() for p in gp['params']):>11,} 个参数  例：{gp['names'][0]}")
        opt = torch.optim.AdamW([{k: v for k, v in gp.items() if k not in ("names", "name")} for gp in groups])
        hist = []
        t0 = time.time()
        for step in range(1, a.overfit_steps + 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(ids, mask, au)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            hist.append(loss.item())
            if step == 1 or step % 5 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                print(f"    step {step:3d}  loss {loss.item():.4f}  batch acc {acc:.2f}")
        ok = hist[-1] < 0.2 * hist[0]
        w = model.layer_mix.weights().detach().cpu()
        print(f"  loss {hist[0]:.3f} → {hist[-1]:.3f}（{time.time() - t0:.1f}s）"
              f"{' ✅ 能把一个 batch 背下来，训练链路没接错' if ok else ' ⚠️ 没降下来，检查学习率 / 梯度'}")
        print(f"  训练后层权重（只训了一个 batch，参考而已）：最大 {w.max():.4f}（第 {int(w.argmax())} 层），"
              f"最小 {w.min():.4f}（第 {int(w.argmin())} 层）")
        checks["overfit"] = ok
        rep["overfit"] = dict(steps=a.overfit_steps, loss_first=round(hist[0], 4), loss_last=round(hist[-1], 4),
                              layer_weights=[round(x, 4) for x in w.tolist()])
        del opt
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------- 6. 文本基线换到 dev 1108 口径 ----------
    print("\n========== 6. 文本基线在多模态同一口径（dev 1108 条）下重算 ==========")
    dev_full = load_split("dev").reset_index(drop=True)
    rows = tables["dev"]["row_csv"].to_numpy()                     # 1108 条在原 1109 行 CSV 里的行号
    gold_full = dev_full["label"].to_numpy()
    pt, js = OUT_DIR / "text_best.pt", OUT_DIR / "text_best.json"
    meta = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
    if meta:
        print(f"  text_best.json：weights={meta.get('weights')} seed={meta.get('seed')} epoch={meta.get('epoch')} "
              f"dev wF1 {meta.get('dev_wf1')}" + ("" if (meta.get("weights"), meta.get("seed")) == ("none", 42)
                                                  else "  ⚠️ 不是 none/seed 42，和日志里定的文本基线不一致"))
    if pt.exists():
        tm = TextClassifier().to(device)
        tm.load_state_dict(torch.load(pt, map_location=device))
        dl = DataLoader(TextDataset(dev_full, tok, a.max_length), batch_size=64, shuffle=False,
                        collate_fn=partial(collate, pad_id=tok.pad_token_id))
        probs, gold = predict(tm, dl, device, amp)
        del tm
        assert (gold == gold_full).all()
        src = "text_best.pt 重新推理"
    else:
        p = OUT_DIR / "day6_dev_probs.npy"
        if not p.exists():
            raise SystemExit("找不到 text_best.pt 也找不到 day6_dev_probs.npy，没法重算文本基线")
        probs = np.load(p)
        src = "day6_dev_probs.npy"
    assert len(probs) == len(dev_full), f"概率 {len(probs)} 行，dev CSV {len(dev_full)} 行"
    m1109 = metrics(probs, gold_full)
    m1108 = metrics(probs[rows], gold_full[rows])
    keep = ~tables["dev"]["shared_audio"].to_numpy()
    m1106 = metrics(probs[rows][keep], gold_full[rows][keep])
    neutral = np.zeros((len(rows), K)); neutral[:, LABELS.index("neutral")] = 1
    mneu = metrics(neutral, gold_full[rows])
    print(f"  来源：{src}")
    print(f"  {'口径':28s} {'n':>5} {'acc':>7} {'wF1':>7} {'mF1':>7}")
    for name, m, n in [("dev 全部（Day 6 口径）", m1109, len(gold_full)),
                       ("dev 有音频（多模态口径）", m1108, len(rows)),
                       ("  └ 再去掉 shared_audio", m1106, int(keep.sum())),
                       ("全猜 neutral（多模态口径）", mneu, len(rows))]:
        print(f"  {name:28s} {n:>5} {m['acc']:>7.4f} {m['wf1']:>7.4f} {m['mf1']:>7.4f}")
    if meta:
        d = abs(m1109["wf1"] - meta["dev_wf1"])
        print(f"  dev 全部的 wF1 与 text_best.json 记录相差 {d:.4f}"
              f"{'（一致 ✅）' if d < 2e-3 else '（⚠️ 对不上，text_best.pt 可能被覆盖过）'}")
        checks["baseline_reproduced"] = d < 2e-3
    print("  → Day 12 / 13 和多模态比的文本基线用「dev 有音频」这一行")
    rep["text_baseline_dev1108"] = dict(
        source=src, n=len(rows), acc=round(m1108["acc"], 4), wf1=round(m1108["wf1"], 4), mf1=round(m1108["mf1"], 4),
        per_class_f1={l: round(v, 4) for l, v in zip(LABELS, m1108["per_class"])},
        full_1109=dict(acc=round(m1109["acc"], 4), wf1=round(m1109["wf1"], 4), mf1=round(m1109["mf1"], 4)),
        no_shared_1106=dict(acc=round(m1106["acc"], 4), wf1=round(m1106["wf1"], 4), mf1=round(m1106["mf1"], 4)),
        all_neutral=dict(acc=round(mneu["acc"], 4), wf1=round(mneu["wf1"], 4), mf1=round(mneu["mf1"], 4)))

    # ---------- 7. 可选：wandb ----------
    if a.wandb:
        import wandb
        run = wandb.init(project="emotion-ai", name="day10-fusion-setup", job_type="setup",
                         config=dict(vars(a), fusion="late/concat", hidden=256, audio_layers="learned weighted sum (13)"))
        wandb.log({"text_baseline_dev1108_wf1": rep["text_baseline_dev1108"]["wf1"],
                   **({"overfit_loss_last": rep["overfit"]["loss_last"]} if "overfit" in rep else {})})
        print(f"\nwandb run：{run.url}")
        wandb.finish()

    # ---------- 保存 ----------
    rep["checks"] = checks
    if device == "cuda":
        rep["gpu_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print(f"\n显存峰值 {rep['gpu_peak_GB']} GB")
    path = OUT_DIR / "day10_check.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"已保存 {path}")
    print("\n检查结果：" + "  ".join(f"{k} {'✅' if v else '⚠️'}" for k, v in checks.items()))
    print("Day 10 完成 ✅" if all(checks.values()) else "有检查没通过 ⚠️ 看上面的输出")


if __name__ == "__main__":
    main()
