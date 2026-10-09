"""
Day 9 补充检查 — 找出音频 embedding 完全相同的样本（同一段 wav 被两个 key 用了）

用法：python src\\check_dups_day9.py
输出：results\\day9_audio_duplicates.csv（每个重复组一行一个样本）
看三件事：
  1. 是否跨 split（train 和 dev/test 共用同一段音频 → 泄漏）
  2. 组内标签是否冲突
  3. 组内文本是否相同（不同 → 音频切错，这段音频不是该句的）
"""
import hashlib

import pandas as pd

from audio_day9 import load_audio_feats
from roberta_day4 import LABELS, OUT_DIR, load_split

parts = []
for s in ["train", "dev", "test"]:
    emb, k = load_audio_feats(s)
    k["split"] = s
    k["h"] = [hashlib.md5(e.tobytes()).hexdigest()[:12] for e in emb[:, -1]]
    t = load_split(s)[["Dialogue_ID", "Utterance_ID", "text", "Speaker"]]
    parts.append(k.merge(t, on=["Dialogue_ID", "Utterance_ID"], how="left"))
k = pd.concat(parts, ignore_index=True)
k["emotion"] = k.label.map(lambda i: LABELS[i])
k["key"] = "dia" + k.Dialogue_ID.astype(str) + "_utt" + k.Utterance_ID.astype(str)

dup = k[k.duplicated("h", keep=False)].sort_values(["h", "split", "Dialogue_ID", "Utterance_ID"])
groups = dup.groupby("h")
print(f"重复组 {groups.ngroups} 个，涉及 {len(dup)} 条样本（按 split：{dup.split.value_counts().to_dict()}）")

cross = [h for h, g in groups if g.split.nunique() > 1]
conflict = [h for h, g in groups if g.label.nunique() > 1]
difftext = [h for h, g in groups if g.text.str.lower().str.strip().nunique() > 1]
print(f"跨 split 的组：{len(cross)}   标签冲突的组：{len(conflict)}   文本不同的组：{len(difftext)}")

for h, g in groups:
    flags = [f for f, lst in (("跨split", cross), ("标签冲突", conflict), ("文本不同", difftext)) if h in lst]
    print(f"\n[{h}] {len(g)} 条  {' '.join(flags)}")
    for r in g.itertuples():
        print(f"  {r.split:5s} {r.key:16s} {r.duration_s:5.2f}s {r.emotion:8s} {r.Speaker}: {str(r.text)[:60]!r}")

cols = ["h", "split", "key", "Dialogue_ID", "Utterance_ID", "duration_s", "emotion", "Speaker", "text"]
dup[cols].to_csv(OUT_DIR / "day9_audio_duplicates.csv", index=False, encoding="utf-8-sig")
print("\n已保存 results\\day9_audio_duplicates.csv")
