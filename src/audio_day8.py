"""
Day 8 — Wav2Vec 2.0：加载模型 → 一条 MELD 音频 → 13 层 hidden states → mean pooling → (1, 768)

用法（PowerShell，cd F:\\condaEnv\\multimodel-emotion，conda activate emotion_ai）：
    python src\\audio_day8.py                                        # 今日全部检查（默认 facebook/wav2vec2-base，首次下载约 360MB 到 F:\\hf_cache）
    python src\\audio_day8.py --file dia0_utt3.wav                   # 换一条 train 音频
    python src\\audio_day8.py --model facebook/wav2vec2-base-960h    # 计划原版（ASR 微调版），对比用
    python src\\audio_day8.py --model microsoft/wavlm-base-plus      # 备选模型
    python src\\audio_day8.py --skip-durations                       # 跳过全量时长统计

做的事：
  1. 加载 feature extractor + 模型，打印关键配置（层数、feat_extract_norm、是否该传 attention_mask）
  2. 读一条 16kHz wav → input_values → last_hidden_state (1, T, 768)，T ≈ 50 帧/秒（卷积总步长 320 = 20ms）
  3. output_hidden_states=True 拿全部 13 层 → 按有效帧数做 mean pooling → 最后一层 (1, 768)，全部层 (1, 13, 768)
  4. 批处理一致性检查：4 条不等长音频，单条算 vs 补零成 batch 算（传 / 不传 attention_mask），比较余弦相似度
     → 决定 Day 9 能不能直接 batch 提取
  5. 所有 split 的 wav 时长统计 + 按 (Dialogue_ID, Utterance_ID) 核对缺失 → 决定截断上限，估算 Day 9 全量用时
  6. 对照（计划里的备用方案）：librosa MFCC(40) 按时间取平均 → (40,)
输出：
  results\\day8_check.json      形状、用时、一致性、时长分位数
  results\\day8_durations.csv   每条样本的 split / Dialogue_ID / Utterance_ID / label / 时长（缺 wav 为空）
  results\\day8_sample_emb.npy  示例音频的 (13, 768) float16（.npy 被 .gitignore 排除）
加载时如果提示 "Some weights ... were not used"（quantizer / project_q / project_hid 等）是正常的：
那是预训练用的对比学习头，提特征用不到。
"""
import argparse
import json
import os
import time
from pathlib import Path

# 必须在 import transformers 之前设置
os.environ.setdefault("HF_HOME", r"F:\hf_cache")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from roberta_day4 import DATA, LABELS, OUT_DIR, load_split
from whisper_wer import SR, load_wav

SPLITS = ["train", "dev", "test"]


# ---------------------------------------------------------------- 工具函数
def wav_table(split):
    """CSV 每一行对应的 wav 路径（按 key 拼，不按行号）。"""
    df = load_split(split)
    df["wav"] = [DATA / "audio" / split / f"dia{d}_utt{u}.wav"
                 for d, u in zip(df.Dialogue_ID, df.Utterance_ID)]
    return df


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def masked_mean(h, n_frames):
    """h: (B, L, T, D)，n_frames: (B,) 每条的有效帧数 → (B, L, D)。补零的帧不参与平均。"""
    T = h.shape[2]
    m = (torch.arange(T, device=h.device)[None, :] < n_frames[:, None].to(h.device)).to(h.dtype)  # (B, T)
    m = m[:, None, :, None]                                                                         # (B, 1, T, 1)
    return (h * m).sum(2) / m.sum(2)


@torch.no_grad()
def embed(model, fe, wavs, device, use_mask=False):
    """一组 16kHz 波形 → (B, 13, 768) 的逐层 mean pooling 向量。"""
    inp = fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
    kw = {"attention_mask": inp.attention_mask.to(device)} if use_mask else {}
    out = model(inp.input_values.to(device), output_hidden_states=True, **kw)
    h = torch.stack(out.hidden_states, dim=1).float()                                # (B, 13, T, 768)
    n_frames = model._get_feat_extract_output_lengths(torch.tensor([len(w) for w in wavs]))
    return masked_mean(h, n_frames)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--split", default="train", choices=SPLITS)
    ap.add_argument("--file", help="指定一条音频，如 dia0_utt3.wav")
    ap.add_argument("--skip-durations", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  HF_HOME={os.environ['HF_HOME']}  model={a.model}")
    OUT_DIR.mkdir(exist_ok=True)
    rep = dict(model=a.model, device=device)

    # ---------- 1. 加载 ----------
    print("\n========== 1. 加载 feature extractor + 模型 ==========")
    t0 = time.time()
    fe = AutoFeatureExtractor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval()  # eval()：关掉 dropout 和训练时的时间遮挡
    cfg = model.config
    norm = getattr(cfg, "feat_extract_norm", "?")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"用时 {time.time() - t0:.1f}s，模型类 {type(model).__name__}，参数量 {n_params:.1f}M")
    print(f"hidden_size {cfg.hidden_size}，Transformer {cfg.num_hidden_layers} 层 → hidden_states 共 "
          f"{cfg.num_hidden_layers + 1} 个（第 0 个是 CNN 特征投影 + 位置卷积后的输出）")
    print(f"feat_extract_norm = {norm}，feature extractor: return_attention_mask = "
          f"{getattr(fe, 'return_attention_mask', '?')}，do_normalize = {getattr(fe, 'do_normalize', '?')}")
    if norm == "group":
        print("  → group norm 版本：HF 文档建议 batch 时不传 attention_mask、只补零；补零到底有没有影响，看第 4 步")
    rep.update(params_M=round(n_params, 1), n_hidden_states=cfg.num_hidden_layers + 1, feat_extract_norm=norm)

    # ---------- 2. 读一条音频 ----------
    print("\n========== 2. 读一条 MELD 音频 ==========")
    df = wav_table(a.split)
    have = df[df.wav.map(Path.exists)]
    sel = have[have.wav.map(lambda p: p.name) == a.file] if a.file else have
    if sel.empty:
        raise SystemExit(f"{a.split} 里找不到 {a.file or '任何 wav'}")
    r = sel.iloc[0]
    y = load_wav(r.wav)
    dur = len(y) / SR
    print(f"{r.wav.name}  [{r.Emotion}]  {r.Speaker}: {r.text!r}")
    print(f"时长 {dur:.2f}s，{len(y)} 个采样点，采样率 {SR}，取值范围 [{y.min():.3f}, {y.max():.3f}]")

    # ---------- 3. forward + mean pooling ----------
    print("\n========== 3. forward → hidden states → mean pooling ==========")
    inp = fe(y, sampling_rate=SR, return_tensors="pt")
    x = inp.input_values.to(device)
    print(f"input_values {tuple(x.shape)}，均值 {x.mean():.3f}，标准差 {x.std():.3f}（do_normalize 后应约为 0 / 1）")
    with torch.no_grad():
        model(x)  # 预热（第一次调用含 CUDA 初始化，不计时）
        sync(device)
        t0 = time.time()
        out = model(x, output_hidden_states=True)
        sync(device)
    secs = time.time() - t0
    last = out.last_hidden_state                                   # (1, T, 768)
    T = last.shape[1]
    n_frames = model._get_feat_extract_output_lengths(torch.tensor([len(y)]))
    print(f"last_hidden_state {tuple(last.shape)}：{T} 帧 / {dur:.2f}s = {T / dur:.1f} 帧每秒；"
          f"按公式算的帧数 {int(n_frames)}")
    print(f"hidden_states 个数 {len(out.hidden_states)}，每个 {tuple(out.hidden_states[0].shape)}")

    h = torch.stack(out.hidden_states, dim=1).float()             # (1, 13, T, 768)
    pooled = masked_mean(h, n_frames)                              # (1, 13, 768)
    audio_emb = pooled[:, -1]                                      # (1, 768)
    same = torch.allclose(audio_emb, last.float().mean(1), atol=1e-4)
    print(f"audio_embedding（最后一层 mean pooling）{tuple(audio_emb.shape)}  "
          f"{'✅' if audio_emb.shape == (1, cfg.hidden_size) else '⚠️'}")
    print(f"全部层 pooled {tuple(pooled.shape)}；单条无补零时 masked_mean == 直接 .mean(1)：{'一致 ✅' if same else '不一致 ⚠️'}")
    cos_to_last = F.cosine_similarity(pooled[0], pooled[0, -1:], dim=-1).tolist()
    print("各层与最后一层的余弦相似度：", " ".join(f"{c:.2f}" for c in cos_to_last))
    print(f"推理用时 {secs * 1000:.0f} ms（{dur:.1f}s 音频）")
    if device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    np.save(OUT_DIR / "day8_sample_emb.npy", pooled[0].cpu().numpy().astype(np.float16))
    rep.update(sample=r.wav.name, sample_emotion=r.Emotion, sample_dur_s=round(dur, 2), frames=T,
               frames_per_s=round(T / dur, 2), last_hidden_state=list(last.shape),
               audio_embedding=list(audio_emb.shape), all_layers=list(pooled.shape), infer_ms=round(secs * 1000, 1))

    # ---------- 4. 批处理一致性 ----------
    print("\n========== 4. 单条 vs 补零 batch 的一致性（为 Day 9 批量提取做准备）==========")
    four = have.sample(n=4, random_state=0)
    wavs = [load_wav(p) for p in four.wav]
    single, t_single = [], []
    for w in wavs:
        sync(device); t0 = time.time()
        single.append(embed(model, fe, [w], device))
        sync(device); t_single.append(time.time() - t0)
    single = torch.cat(single)                                     # (4, 13, 768)
    print("4 条音频时长：", ", ".join(f"{len(w) / SR:.1f}s" for w in wavs),
          f"；单条平均用时 {np.mean(t_single) * 1000:.0f} ms")
    consistency = {}
    for use_mask in (False, True):
        b = embed(model, fe, wavs, device, use_mask)
        cos = F.cosine_similarity(single, b, dim=-1)               # (4, 13)
        tag = "传 attention_mask" if use_mask else "不传 attention_mask"
        print(f"  {tag:18s} 最后一层余弦 " + " ".join(f"{c:.4f}" for c in cos[:, -1].tolist())
              + f"   全部层最小 {cos.min():.4f}")
        consistency["with_mask" if use_mask else "no_mask"] = dict(
            last_layer=[round(c, 5) for c in cos[:, -1].tolist()], min_all_layers=round(float(cos.min()), 5))
    best = max(v["min_all_layers"] for v in consistency.values())
    if best > 0.999:
        print("  → 补零几乎不影响结果，Day 9 可以直接 batch（按时长排序分桶，减少补零）")
    else:
        print("  → 补零会改变结果（最短的那条通常差得最多）。Day 9 建议一条一条提（4090 上也很快），"
              "或只把时长几乎相同的放一个 batch")
    rep["batch_consistency"] = consistency
    rep["single_clip_ms_mean"] = round(float(np.mean(t_single)) * 1000, 1)

    # ---------- 5. 时长统计 + 缺失核对 ----------
    if not a.skip_durations:
        print("\n========== 5. 全部 split 的 wav 时长 + 缺失核对 ==========")
        t0 = time.time()
        rows = []
        for s in SPLITS:
            d = wav_table(s)
            for dia, utt, lab, p in zip(d.Dialogue_ID, d.Utterance_ID, d.label, d.wav):
                rows.append(dict(split=s, Dialogue_ID=dia, Utterance_ID=utt, label=LABELS[lab],
                                 duration_s=round(sf.info(str(p)).duration, 3) if p.exists() else np.nan))
        dur_df = pd.DataFrame(rows)
        dur_df.to_csv(OUT_DIR / "day8_durations.csv", index=False)
        print(f"读 {len(dur_df)} 个文件头用时 {time.time() - t0:.1f}s")
        rep["durations"] = {}
        for s, g in dur_df.groupby("split", sort=False):
            miss = g[g.duration_s.isna()]
            v = g.duration_s.dropna()
            q = {p: round(float(np.percentile(v, p)), 1) for p in (50, 90, 95, 99)}
            print(f"[{s}] {len(g)} 条，缺 wav {len(miss)}"
                  + (f"（{', '.join(f'dia{x}_utt{y}' for x, y in zip(miss.Dialogue_ID, miss.Utterance_ID))[:120]}）"
                     if len(miss) else "")
                  + f"；时长 中位数 {q[50]}s / 90% {q[90]}s / 95% {q[95]}s / 99% {q[99]}s / 最长 {v.max():.1f}s；"
                  f">10s {int((v > 10).sum())} 条，>20s {int((v > 20).sum())} 条，<0.5s {int((v < 0.5).sum())} 条")
            rep["durations"][s] = dict(n=len(g), missing=len(miss), pct=q, max=round(float(v.max()), 1),
                                       over_10s=int((v > 10).sum()), over_20s=int((v > 20).sum()))
        longest = dur_df.nlargest(3, "duration_s")
        print("最长的 3 条：", "; ".join(f"{r.split} dia{r.Dialogue_ID}_utt{r.Utterance_ID} {r.duration_s:.1f}s"
                                     for r in longest.itertuples()))
        print("按情绪的平均时长（train）：", ", ".join(
            f"{k} {v:.2f}s" for k, v in dur_df[dur_df.split == "train"].groupby("label").duration_s.mean().items()))
        n_all = int(dur_df.duration_s.notna().sum())
        est = n_all * np.mean(t_single) / 60
        print(f"粗估 Day 9 全量提取（{n_all} 条、逐条、不含读盘）：约 {est:.0f} 分钟")
        rep["estimate_full_extract_min"] = round(float(est), 1)

    # ---------- 6. MFCC 对照 ----------
    print("\n========== 6. 对照：librosa MFCC(40) ==========")
    try:
        import librosa
        mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)          # (40, 帧数)
        print(f"MFCC {mf.shape}（默认 hop 512 = 32ms，约 {mf.shape[1] / dur:.0f} 帧每秒）→ 按时间平均 {mf.mean(1).shape}")
        print("  这是计划里 Wav2Vec 太慢时的备用方案；4090 上用不到，但 Day 13 可以当一个弱基线")
    except Exception as e:
        print(f"(MFCC 跳过：{e})")

    path = OUT_DIR / "day8_check.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存 {path}、day8_sample_emb.npy" + ("" if a.skip_durations else "、day8_durations.csv"))
    print("\nDay 8 完成 ✅")


if __name__ == "__main__":
    main()
