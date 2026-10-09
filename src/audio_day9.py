"""
Day 9 — 全量提取 Wav2Vec 2.0 音频特征：train + dev + test，每条 13 层 mean pooling，float16 存 .npz

用法（PowerShell，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python src\\audio_day9.py --limit 20                    # 冒烟测试：每个 split 前 20 条 → *_smoke.npz，不覆盖正式文件
    python src\\audio_day9.py                               # 正式：三个 split 全量（4090 约 3–5 分钟）+ 检查
    python src\\audio_day9.py --probe                       # 提完后再做 13 层线性探针（dev weighted F1，约几分钟）
    python src\\audio_day9.py --skip-extract --probe        # 已经提好了，只做检查 + 探针
    python src\\audio_day9.py --plot                        # 可选：每类情绪一条语谱图 → results\\day9_spectrograms.png
    python src\\audio_day9.py --model microsoft/wavlm-base-plus   # 可选：再提一份 WavLM 做对比（文件名带模型名，不冲突）

做法（Day 8 的结论）：
  - 逐条提取（batch=1），不补零、不传 attention_mask（Day 8：补零会让短句的 embedding 明显变样）
  - model.eval() + no_grad，fp32 推理（和 day8_sample_emb.npy 同一算法，方便比对）
  - 超过 --max-sec（默认 20s）的片段只取前 20 秒，标记 truncated（MELD 切分错误的 305s / 235s / 41s 等）
  - 不足 400 个采样点（25ms，卷积前端的最小窗口）补零到 400，标记 padded
  - 缺 wav 的行（train dia125_utt3、dev dia110_utt7）不写入，多模态实验按 key 对齐时自然丢弃
输出（.npz 被 .gitignore 排除，不进仓库）：
  results\\audio_{模型名}_{split}.npz   emb (N, 13, 768) float16、dialogue_id、utterance_id、label、
                                         duration_s（原始时长）、truncated、padded、model
  results\\day9_extract.json            条数 / 缺失 / 截断 / 用时 / 文件大小 / 检查结果
  results\\day9_layer_probe.csv / .png  （--probe）每层 logistic regression 在 dev 上的 acc / weighted F1 / macro F1
  results\\day9_spectrograms.png        （--plot）
Day 10 读特征：from audio_day9 import load_audio_feats
"""
import argparse
import json
import os
import time
from pathlib import Path

# 必须在 import transformers 之前设置
os.environ.setdefault("HF_HOME", r"F:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from audio_day8 import SPLITS, embed, wav_table
from roberta_day4 import LABELS, OUT_DIR
from whisper_wer import SR, load_wav

MIN_SAMPLES = 400  # wav2vec2 第一层卷积 kernel=400（25ms），更短的输入得到 0 帧


def feat_path(model_name, split, smoke=False):
    return OUT_DIR / f"audio_{model_name.split('/')[-1]}_{split}{'_smoke' if smoke else ''}.npz"


def load_audio_feats(split, model_name="facebook/wav2vec2-base", layer=None):
    """Day 10 起用：返回 (emb, keys)。emb 为 (N, 13, 768) float16，给 layer 则为 (N, 768)；
    keys 是 DataFrame[Dialogue_ID, Utterance_ID, label, duration_s, truncated, padded]，行顺序和 emb 一致。
    和文本对齐请按 (Dialogue_ID, Utterance_ID) merge，不要按行号。"""
    z = np.load(feat_path(model_name, split))
    emb = z["emb"] if layer is None else z["emb"][:, layer]
    keys = pd.DataFrame(dict(Dialogue_ID=z["dialogue_id"], Utterance_ID=z["utterance_id"], label=z["label"],
                             duration_s=z["duration_s"], truncated=z["truncated"], padded=z["padded"]))
    return emb, keys


# ---------------------------------------------------------------- 提取
def extract_split(model, fe, model_name, split, device, max_sec, limit=None):
    df = wav_table(split)
    n_csv = len(df)
    exists = df.wav.map(Path.exists)
    missing = [f"dia{d}_utt{u}" for d, u in zip(df.Dialogue_ID[~exists], df.Utterance_ID[~exists])]
    df = df[exists].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    n = len(df)
    L, D = model.config.num_hidden_layers + 1, model.config.hidden_size
    print(f"\n[{split}] CSV {n_csv} 条，缺 wav {len(missing)}（{', '.join(missing) or '无'}），本次提取 {n} 条")

    emb = np.zeros((n, L, D), np.float16)
    dur = np.zeros(n, np.float32)
    trunc = np.zeros(n, bool)
    padded = np.zeros(n, bool)
    max_len = int(max_sec * SR)
    t0 = time.time()
    t_model = 0.0
    for i, p in enumerate(df.wav):
        y = load_wav(p)
        dur[i] = len(y) / SR
        if len(y) > max_len:
            y, trunc[i] = y[:max_len], True
        if len(y) < MIN_SAMPLES:
            y, padded[i] = np.pad(y, (0, MIN_SAMPLES - len(y))), True
        tm = time.time()
        e = embed(model, fe, [y], device)[0]                     # (13, 768) float32
        emb[i] = e.cpu().numpy().astype(np.float16)
        t_model += time.time() - tm
        if (i + 1) % 1000 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"  {i + 1}/{n}  已用 {el / 60:.1f} 分钟，预计还需 {el / (i + 1) * (n - i - 1) / 60:.1f} 分钟")
    secs = time.time() - t0

    path = feat_path(model_name, split, smoke=bool(limit))
    np.savez(path, emb=emb, dialogue_id=df.Dialogue_ID.to_numpy(), utterance_id=df.Utterance_ID.to_numpy(),
             label=df.label.to_numpy(), duration_s=dur, truncated=trunc, padded=padded, model=np.array(model_name))
    print(f"  用时 {secs / 60:.1f} 分钟（模型 {t_model / 60:.1f} 分钟，其余是读 wav），"
          f"平均 {secs / max(n, 1) * 1000:.0f} ms/条 → {path.name}")
    if trunc.any():
        print("  截断：", ", ".join(f"dia{d}_utt{u}({s:.0f}s)" for d, u, s in
                                   zip(df.Dialogue_ID[trunc], df.Utterance_ID[trunc], dur[trunc])))
    if padded.any():
        print(f"  补零到 {MIN_SAMPLES} 采样点：{int(padded.sum())} 条")
    return dict(n_csv=n_csv, missing=missing, n=n, truncated=int(trunc.sum()), padded=int(padded.sum()),
                minutes=round(secs / 60, 2), ms_per_clip=round(secs / max(n, 1) * 1000, 1))


# ---------------------------------------------------------------- 检查
def verify(model_name, split, smoke=False):
    path = feat_path(model_name, split, smoke)
    z = np.load(path)
    emb = z["emb"]
    keys = list(zip(z["dialogue_id"].tolist(), z["utterance_id"].tolist()))
    df = wav_table(split)
    expected = int(df.wav.map(Path.exists).sum())
    res = dict(
        file=path.name, shape=list(emb.shape), dtype=str(emb.dtype), size_MB=round(path.stat().st_size / 1e6, 1),
        finite=bool(np.isfinite(emb).all()), unique_keys=len(set(keys)) == len(keys),
        count_ok=smoke or len(keys) == expected, expected=expected,
        labels_ok=bool(np.isin(z["label"], np.arange(len(LABELS))).all()),
        abs_max=float(np.abs(emb.astype(np.float32)).max()),
    )
    # 标签和 CSV 按 key 再核对一次
    lab = dict(zip(zip(df.Dialogue_ID, df.Utterance_ID), df.label))
    res["labels_match_csv"] = all(lab[k] == l for k, l in zip(keys, z["label"].tolist()))
    # 近似重复：同一个 embedding 出现两次（说明读错文件）
    last = emb[:, -1].astype(np.float32)
    res["exact_duplicate_rows"] = int(len(last) - len(np.unique(last, axis=0)))
    ok = res["finite"] and res["unique_keys"] and res["count_ok"] and res["labels_ok"] and res["labels_match_csv"]
    print(f"[{split}] {path.name}: shape {tuple(emb.shape)} {emb.dtype}，{res['size_MB']} MB；"
          f"NaN/inf {'无' if res['finite'] else '有 ⚠️'}；key 唯一 {'✅' if res['unique_keys'] else '⚠️'}；"
          f"条数 {len(keys)}" + ("" if smoke else f" / 应为 {expected} {'✅' if res['count_ok'] else '⚠️'}")
          + f"；标签与 CSV 一致 {'✅' if res['labels_match_csv'] else '⚠️'}；"
          f"完全重复的行 {res['exact_duplicate_rows']}；|值| 最大 {res['abs_max']:.1f}")
    res["ok"] = bool(ok)
    return res


def compare_day8(model_name):
    """dia0_utt0 应和 Day 8 保存的 day8_sample_emb.npy 一致（同模型、同算法）。"""
    ref_path = OUT_DIR / "day8_sample_emb.npy"
    chk = OUT_DIR / "day8_check.json"
    if not ref_path.exists():
        print("（没有 day8_sample_emb.npy，跳过比对）")
        return None
    if chk.exists() and json.loads(chk.read_text(encoding="utf-8")).get("model") != model_name:
        print("（day8_sample_emb.npy 是另一个模型提的，跳过比对）")
        return None
    emb, keys = load_audio_feats("train", model_name)
    idx = keys.index[(keys.Dialogue_ID == 0) & (keys.Utterance_ID == 0)]
    if len(idx) == 0:
        return None
    a = emb[idx[0]].astype(np.float32)
    b = np.load(ref_path).astype(np.float32)
    cos = F.cosine_similarity(torch.from_numpy(a), torch.from_numpy(b), dim=-1)
    diff = float(np.abs(a - b).max())
    print(f"dia0_utt0 vs day8_sample_emb.npy：13 层余弦最小 {cos.min():.6f}，最大绝对差 {diff:.4f} "
          f"{'✅ 一致' if cos.min() > 0.9999 else '⚠️ 不一致'}（float16 + GPU 浮点顺序，差 1e-2 以内算一致）")
    return dict(cos_min=round(float(cos.min()), 6), max_abs_diff=round(diff, 5))


# ---------------------------------------------------------------- 可选：逐层线性探针
def layer_probe(model_name):
    """每一层单独做 标准化 + logistic regression（train 训练，dev 评估）。
    只是看"哪一层情绪信息多"的快速探针，不是 Day 13 的 audio-only 正式结果。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    Xtr, ktr = load_audio_feats("train", model_name)
    Xdv, kdv = load_audio_feats("dev", model_name)
    ytr, ydv = ktr.label.to_numpy(), kdv.label.to_numpy()
    variants = [(str(l), l) for l in range(Xtr.shape[1])] + [("mean13", None)]
    rows = []
    print(f"\n逐层线性探针：train {len(ytr)} → dev {len(ydv)}，logistic regression（不加 class weights，和文本基线同口径）")
    for name, l in variants:
        t0 = time.time()
        a = Xtr[:, l] if l is not None else Xtr.astype(np.float32).mean(1)
        b = Xdv[:, l] if l is not None else Xdv.astype(np.float32).mean(1)
        sc = StandardScaler().fit(a.astype(np.float32))
        clf = LogisticRegression(max_iter=3000, C=0.1)
        clf.fit(sc.transform(a.astype(np.float32)), ytr)
        pred = clf.predict(sc.transform(b.astype(np.float32)))
        r = dict(layer=name, dev_acc=accuracy_score(ydv, pred),
                 dev_wF1=f1_score(ydv, pred, average="weighted"), dev_mF1=f1_score(ydv, pred, average="macro"),
                 pred_neutral_frac=float((pred == LABELS.index("neutral")).mean()), secs=time.time() - t0)
        rows.append(r)
        print(f"  layer {name:>6s}  acc {r['dev_acc']:.3f}  wF1 {r['dev_wF1']:.3f}  mF1 {r['dev_mF1']:.3f}  "
              f"预测 neutral 占比 {r['pred_neutral_frac']:.2f}  ({r['secs']:.0f}s)")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT_DIR / "day9_layer_probe.csv", index=False)
    best = res.loc[res.dev_wF1.idxmax()]
    print(f"最好的层：{best.layer}（dev wF1 {best.dev_wF1:.3f}）；参考：全猜 neutral 0.252，文本基线 0.604")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = res[res.layer != "mean13"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(per.layer.astype(int), per.dev_wF1, "o-", label="weighted F1")
    ax.plot(per.layer.astype(int), per.dev_mF1, "s-", label="macro F1")
    ax.axhline(0.252, ls=":", c="gray", label="all-neutral wF1")
    ax.set_xlabel("wav2vec2 hidden state (0 = CNN features)")
    ax.set_ylabel("dev score (linear probe)")
    ax.set_title(f"Per-layer linear probe — {model_name.split('/')[-1]}")
    ax.set_xticks(range(len(per)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day9_layer_probe.png", dpi=150)
    print("已保存 results\\day9_layer_probe.csv、day9_layer_probe.png")
    return res.round(4).to_dict(orient="records")


# ---------------------------------------------------------------- 可选：语谱图
def plot_spectrograms():
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    df = wav_table("train")
    df = df[df.wav.map(Path.exists)]
    fig, axes = plt.subplots(len(LABELS), 1, figsize=(9, 2.1 * len(LABELS)))
    for ax, (lab_id, lab) in zip(axes, enumerate(LABELS)):
        g = df[df.label == lab_id].head(200)
        d = g.wav.map(lambda p: sf.info(str(p)).duration)
        g = g[(d >= 2.0) & (d <= 4.0)] if ((d >= 2.0) & (d <= 4.0)).any() else g
        r = g.sample(n=1, random_state=0).iloc[0]
        y = load_wav(r.wav)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(S, sr=SR, hop_length=160, x_axis="time", y_axis="hz", ax=ax)
        ax.set_ylim(0, 4000)
        ax.set_title(f"{lab} — {r.wav.name}: {r.text[:70]!r}", fontsize=9)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day9_spectrograms.png", dpi=120)
    print("已保存 results\\day9_spectrograms.png（每类一条 2–4s 的 train 音频，0–4kHz）")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--max-sec", type=float, default=20.0, help="截断上限（秒）")
    ap.add_argument("--limit", type=int, help="冒烟测试：每个 split 只提前 N 条，写 *_smoke.npz")
    ap.add_argument("--skip-extract", action="store_true", help="不提取，只检查已有文件（配合 --probe）")
    ap.add_argument("--probe", action="store_true", help="逐层线性探针（需要 train 和 dev 的正式文件）")
    ap.add_argument("--plot", action="store_true", help="每类情绪一条语谱图")
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    smoke = bool(a.limit)
    rep = dict(model=a.model, max_sec=a.max_sec, min_samples=MIN_SAMPLES, smoke=smoke)

    if not a.skip_extract:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device={device}  HF_HOME={os.environ['HF_HOME']}  model={a.model}  截断 {a.max_sec}s")
        fe = AutoFeatureExtractor.from_pretrained(a.model)
        model = AutoModel.from_pretrained(a.model).to(device).eval()
        rep["device"] = device
        rep["extract"] = {s: extract_split(model, fe, a.model, s, device, a.max_sec, a.limit) for s in a.splits}
        del model
        if device == "cuda":
            print(f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            torch.cuda.empty_cache()

    print("\n========== 检查 ==========")
    rep["verify"] = {s: verify(a.model, s, smoke) for s in a.splits if feat_path(a.model, s, smoke).exists()}
    if not smoke and "train" in rep["verify"]:
        rep["day8_match"] = compare_day8(a.model)
    all_ok = all(v["ok"] for v in rep["verify"].values())
    print("全部检查通过 ✅" if all_ok else "有检查没通过 ⚠️ 看上面的输出")

    if a.probe:
        if smoke:
            print("冒烟模式不做探针")
        else:
            rep["layer_probe"] = layer_probe(a.model)
    if a.plot:
        plot_spectrograms()

    name = "day9_extract_smoke.json" if smoke else "day9_extract.json"
    (OUT_DIR / name).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n已保存 results\\{name}")
    if not smoke and all_ok:
        print("Day 9 完成 ✅")


if __name__ == "__main__":
    main()
