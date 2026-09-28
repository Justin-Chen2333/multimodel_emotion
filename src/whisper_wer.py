"""
Day 3 — Whisper 转录 MELD 音频，并与 CSV 里的 Utterance 标注对比算 WER

先装（Anaconda Prompt，conda activate emotion_ai）：
    python -m pip install -U openai-whisper jiwer
用法（在 F:\\condaEnv\\multimodel-emotion 下）：
    python whisper_wer.py --n 1                  # 今日最低目标：转 1 条并打印
    python whisper_wer.py --n 50                 # 可选任务：随机 50 条 → results\\transcription_results.csv
    python whisper_wer.py --n 50 --split dev     # 换 split
    python whisper_wer.py --n 50 --model base    # 换模型对比（tiny/base/small/medium）
    python whisper_wer.py --all --split test     # 全量：results\\transcripts_test_small.csv，每 200 条存盘，中断后重跑会续上

说明：
- 音频直接用 soundfile 读成 16kHz float32 数组喂给 Whisper，不经过 ffmpeg，
  所以在 Jupyter 里 PATH 不对也不会报 "ffmpeg not found"。
- 参考文本和转录都先过 Whisper 自带的 EnglishTextNormalizer（去标点、统一大小写/数字/缩写），
  这是算 Whisper WER 的标准做法，否则 "Okay." vs "OK" 这种也算错。
- 模型缓存在 F:\\hf_cache\\whisper，不占 C 盘。
"""
import argparse
import re
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper
from whisper.normalizers import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MELD"
OUT_DIR = ROOT / "results"
MODEL_DIR = r"F:\hf_cache\whisper"
SR = 16000

# MELD 的 CSV 里有 Windows-1252 残留字符（如 \x92 其实是撇号），不修会被算成错词
_REPLACE = {"\x91": "'", "\x92": "'", "\u2018": "'", "\u2019": "'",
            "\x93": '"', "\x94": '"', "\u201c": '"', "\u201d": '"',
            "\x85": "...", "\u2026": "...", "\x96": "-", "\x97": "-", "\u2013": "-", "\u2014": "-"}


def _unmojibake(m):
    run = m.group(0)
    b = bytearray()
    for ch in run:
        try:
            b += ch.encode("cp1252")
        except UnicodeEncodeError:
            try:
                b += ch.encode("latin-1")  # \x80 这类 cp1252 编不了的控制字符
            except UnicodeEncodeError:
                return run
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return run  # 不是乱码（比如单独的 \x92），交给下面的 _REPLACE


def fix_text(s):
    # 形如 "itâ€™s" 的乱码：UTF-8 字节被当成 cp1252/latin-1 读了，只修连续的非 ASCII 片段
    s = re.sub(r"[^\x00-\x7f]+", _unmojibake, str(s))
    for a, b in _REPLACE.items():
        s = s.replace(a, b)
    return " ".join(s.split())


_SAME = {"ok": "okay", "yeah": "yes", "yep": "yes", "uh": "", "um": "", "hmm": "", "mm": ""}


def normalize(norm, s):
    """Whisper 官方英文规范化 + 几个口语同义词统一（OK/okay 这种不该算错）"""
    toks = (_SAME.get(t, t) for t in norm(s).split())
    return " ".join(t for t in toks if t and any(c.isalnum() for c in t))  # 去掉 "." 这种纯标点残留


def load_wav(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:  # extract_audio.py 输出已是 16k，这里只是保险
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="转录几条（1 = 取 CSV 第一条已有音频的样本）")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="转录该 split 的全部样本（按 CSV 顺序，支持断点续跑）")
    a = ap.parse_args()

    df = pd.read_csv(DATA / "annotations" / f"{a.split}_sent_emo.csv", encoding="utf-8")
    df["wav"] = [DATA / "audio" / a.split / f"dia{d}_utt{u}.wav"
                 for d, u in zip(df.Dialogue_ID, df.Utterance_ID)]
    have = df[df["wav"].map(Path.exists)]
    print(f"[{a.split}] CSV {len(df)} 条，已转好的 wav {len(have)} 条")
    if have.empty:
        raise SystemExit(f"没有 wav，先跑：python extract_audio.py --split {a.split} --limit 50")
    OUT_DIR.mkdir(exist_ok=True)
    if a.all:
        out_csv = OUT_DIR / f"transcripts_{a.split}_{a.model}.csv"
        done = pd.read_csv(out_csv, keep_default_na=False) if out_csv.exists() else pd.DataFrame()
        done_files = set(done["file"]) if len(done) else set()
        sample = have[~have["wav"].map(lambda p: p.name in done_files)]
        print(f"全量模式：已完成 {len(done_files)} 条，本次待转 {len(sample)} 条 → {out_csv}")
    else:
        done = pd.DataFrame()
        out_csv = OUT_DIR / ("transcription_results.csv" if a.n > 1 else "transcription_one.csv")
        sample = have.head(1) if a.n == 1 else have.sample(n=min(a.n, len(have)), random_state=a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 whisper-{a.model} 到 {device} ...")
    model = whisper.load_model(a.model, device=device, download_root=MODEL_DIR)
    norm = EnglishTextNormalizer()

    rows = []
    t_all = time.time()
    for i, r in enumerate(sample.itertuples(index=False), 1):
        audio = load_wav(r.wav)
        t0 = time.time()
        out = model.transcribe(audio, language="en", fp16=(device == "cuda"),
                               condition_on_previous_text=False)
        hyp = out["text"].strip()
        ref = fix_text(r.Utterance)
        ref_n, hyp_n = normalize(norm, ref), normalize(norm, hyp)
        wer = jiwer.wer(ref_n, hyp_n) if ref_n else float("nan")
        rows.append(dict(file=r.wav.name, dialogue_id=r.Dialogue_ID, utterance_id=r.Utterance_ID, emotion=r.Emotion, speaker=r.Speaker,
                         duration_s=round(len(audio) / SR, 2), ref=ref, hyp=hyp,
                         ref_norm=ref_n, hyp_norm=hyp_n, wer=round(wer, 3),
                         secs=round(time.time() - t0, 2)))
        if a.all and i % 200 == 0:  # 定期存盘，断了也不白跑
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out_csv, index=False, encoding="utf-8-sig")
            el = time.time() - t_all
            print(f"  {i}/{len(sample)}  已用 {el / 60:.1f} 分钟，预计还需 {el / i * (len(sample) - i) / 60:.1f} 分钟")
        elif not a.all and a.n <= 5:
            print(f"\n{r.wav.name}  [{r.Emotion}]  {len(audio) / SR:.1f}s")
            print(f"  MELD   : {ref}")
            print(f"  Whisper: {hyp}")
            print(f"  WER    : {wer:.3f}")
        elif not a.all and i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    res = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    res["ref_norm"] = res["ref_norm"].fillna("").astype(str)
    res["hyp_norm"] = res["hyp_norm"].fillna("").astype(str)
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")  # utf-8-sig：Excel 直接打开不乱码

    ok = res[res.ref_norm.str.len() > 0]
    corpus_wer = jiwer.wer(ok.ref_norm.tolist(), ok.hyp_norm.tolist())
    long_ = ok[ok.ref_norm.str.split().str.len() >= 3]
    print(f"\n共 {len(res)} 条，用时 {time.time() - t_all:.1f}s；corpus WER = {corpus_wer:.3f}")
    if len(res) > 1:
        print(f"参考 ≥3 词的句子（{len(long_)} 条）corpus WER = {jiwer.wer(long_.ref_norm.tolist(), long_.hyp_norm.tolist()):.3f}")
        print(f"逐句 WER 中位数 = {res.wer.median():.3f}，WER=0 的句子 {int((res.wer == 0).sum())} 条")
        print("\n按情绪（corpus WER / 条数）：")
        for emo, g in ok.groupby("emotion"):
            print(f"  {emo:9s} {jiwer.wer(g.ref_norm.tolist(), g.hyp_norm.tolist()):.3f}  ({len(g)})")
        print("\nWER 最高的 5 条：")
        for _, r in res.sort_values("wer", ascending=False).head(5).iterrows():
            print(f"  {r.wer:.2f} {r.file}\n    MELD   : {r.ref}\n    Whisper: {r.hyp}")
    print(f"\n结果已保存：{out_csv}")


if __name__ == "__main__":
    main()
