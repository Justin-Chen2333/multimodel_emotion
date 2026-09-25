"""
Day 4 — RoBERTa tokenizer → [CLS] 向量 → Linear(768, 7) forward pass

用法（Anaconda Prompt，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python roberta_day4.py                                  # 跑完今天全部检查
    python roberta_day4.py --text "Oh my God, you're kidding!"   # 换一句话看 tokenize 结果

做的事：
  1. tokenizer 把一句 MELD 文本变成 input_ids（今日最低目标）
  2. 展示 fix_text 清洗前后的差别（CSV 里有乱码）
  3. 统计 train 全部句子的 token 长度 → 决定 Day 5 的 max_length
  4. RoBERTa 取 [CLS]（RoBERTa 里叫 <s>）向量 (batch, 768) → Linear(768, 7) → (batch, 7) logits
  5. 算一次 CrossEntropy loss 并 backward，确认梯度能传回 RoBERTa（Day 5 训练的前提）
  6. 定下 label→id 顺序和 class weights，存到 results\\label_map.json，之后训练/评估都读这个文件
"""
import argparse
import json
import math
import os
from pathlib import Path

# 必须在 import transformers 之前设置，否则 roberta-base（约 500MB）会下到 C 盘
os.environ.setdefault("HF_HOME", r"F:\hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

from whisper_wer import fix_text  # Day 3 写的乱码清洗

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "MELD"
OUT_DIR = ROOT / "results"
MODEL_NAME = "roberta-base"

# 固定顺序（字母序），全项目统一使用，不要再改
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


class TextClassifier(nn.Module):
    """RoBERTa + 取第 0 个位置（<s>，相当于 BERT 的 [CLS]）+ Linear(768, 7)。Day 5 直接 import 这个类训练。"""

    def __init__(self, n_classes=len(LABELS), dropout=0.1, model_name=MODEL_NAME):
        super().__init__()
        # add_pooling_layer=False：不要 RoBERTa 自带的 pooler（我们自己取 [CLS]）。
        # 加载时可能提示 "Some weights ... were not used: pooler / lm_head"——正常，那些是我们不用的层
        self.encoder = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, L, 768)
        cls = h[:, 0, :]                                                                         # (B, 768)
        return self.head(self.dropout(cls)), cls                                                 # (B, 7), (B, 768)


def load_split(split):
    df = pd.read_csv(DATA / "annotations" / f"{split}_sent_emo.csv", encoding="utf-8")
    df["text"] = df["Utterance"].map(fix_text)
    df["label"] = df["Emotion"].map(LABEL2ID)
    unknown = set(df.loc[df["label"].isna(), "Emotion"])
    assert not unknown, f"CSV 里有未知情绪标签：{unknown}"
    df["label"] = df["label"].astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="自己指定一句话来看 tokenize 结果")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"HF_HOME = {os.environ['HF_HOME']}    device = {device}")
    OUT_DIR.mkdir(exist_ok=True)
    df = load_split("train")

    # ---------- 1. tokenizer ----------
    print("\n========== 1. Tokenizer ==========")
    tok = RobertaTokenizer.from_pretrained(MODEL_NAME)
    text = a.text or df.text.iloc[0]
    enc = tok(text, return_tensors="pt")
    print(f"文本      : {text}")
    print(f"input_ids : {enc['input_ids'].tolist()[0]}   shape={tuple(enc['input_ids'].shape)}")
    print(f"tokens    : {tok.convert_ids_to_tokens(enc['input_ids'][0])}")
    print(f"decode    : {tok.decode(enc['input_ids'][0])}")
    print("说明：<s>=0 在句首（取它的向量当句子表示），</s>=2 在句尾；'Ġ' 表示这个子词前面有空格（BPE 分词）")

    # ---------- 2. 乱码清洗 ----------
    print("\n========== 2. fix_text 清洗 ==========")
    changed = df[df.Utterance.astype(str).str.split().str.join(" ") != df.text]
    print(f"train 里被清洗改动的句子：{len(changed)} / {len(df)}")
    if len(changed):
        r = changed.iloc[0]
        print(f"  原始: {r.Utterance!r}\n  清洗: {r.text!r}")
        print(f"  原始 tokens 数 {len(tok(r.Utterance)['input_ids'])} → 清洗后 {len(tok(r.text)['input_ids'])}")

    # ---------- 3. token 长度 ----------
    print("\n========== 3. train token 长度（含 <s></s>）==========")
    lens = np.array([len(x) for x in tok(df.text.tolist())["input_ids"]])
    pct = {p: int(np.percentile(lens, p)) for p in (50, 90, 95, 99)}
    print(f"中位数 {pct[50]}，90% {pct[90]}，95% {pct[95]}，99% {pct[99]}，最长 {lens.max()}")
    print(f"超过 max_length={a.max_length} 会被截断的：{int((lens > a.max_length).sum())} 条")

    # ---------- 4. forward pass ----------
    print("\n========== 4. [CLS] → Linear(768, 7) forward ==========")
    torch.manual_seed(42)
    model = TextClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.1f}M（RoBERTa-base 约 124M + 分类头 768*7+7）")

    batch = df.sample(n=a.batch, random_state=42)
    enc = tok(batch.text.tolist(), padding=True, truncation=True,
              max_length=a.max_length, return_tensors="pt").to(device)
    labels = torch.tensor(batch.label.values, device=device)
    print(f"batch input_ids {tuple(enc['input_ids'].shape)}（padding 到本 batch 最长句），"
          f"各句真实长度 {enc['attention_mask'].sum(1).tolist()}")

    model.eval()
    with torch.no_grad():
        logits, cls = model(enc["input_ids"], enc["attention_mask"])
    print(f"cls    shape {tuple(cls.shape)}   (应为 ({a.batch}, 768))")
    print(f"logits shape {tuple(logits.shape)}   (应为 ({a.batch}, 7))")
    probs = logits.softmax(-1)[0].tolist()
    print(f"第 1 句：{batch.text.iloc[0]!r}  真实={batch.Emotion.iloc[0]}")
    print("  未训练的概率分布（应接近均匀的 1/7≈0.143）：",
          {ID2LABEL[i]: round(p, 3) for i, p in enumerate(probs)})

    # ---------- 5. loss + backward ----------
    print("\n========== 5. loss + backward ==========")
    model.train()
    logits, _ = model(enc["input_ids"], enc["attention_mask"])
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    print(f"loss = {loss.item():.4f}   （随机分类头的理论值 ln7 = {math.log(7):.4f}，接近即正常）")
    g_head = model.head.weight.grad.norm().item()
    g_emb = model.encoder.embeddings.word_embeddings.weight.grad.norm().item()
    print(f"梯度范数：分类头 {g_head:.4f}，RoBERTa 词嵌入层 {g_emb:.4f}  → 都 >0 说明梯度传通了")
    if device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB（batch={a.batch}）")

    # ---------- 6. label map + class weights ----------
    print("\n========== 6. label→id 与 class weights ==========")
    counts = df.label.value_counts().reindex(range(len(LABELS)), fill_value=0)
    weights = (len(df) / (len(LABELS) * counts)).round(4)  # sklearn 'balanced'：N / (K * n_c)
    for i, l in ID2LABEL.items():
        print(f"  {i} {l:9s} {counts[i]:5d} 条 ({counts[i] / len(df):.1%})  weight={weights[i]}")
    out = dict(labels=LABELS, label2id=LABEL2ID, train_counts=counts.tolist(),
               class_weights_balanced=weights.tolist(), model=MODEL_NAME,
               token_len_percentiles=pct, token_len_max=int(lens.max()), max_length=a.max_length)
    path = OUT_DIR / "label_map.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存 {path}")
    print("\nDay 4 检查全部通过 ✅" if logits.shape == (a.batch, len(LABELS)) else "\n!! logits 维度不对")


if __name__ == "__main__":
    main()
