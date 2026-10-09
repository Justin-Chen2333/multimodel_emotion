"""
Day 15 — Gradio demo 基本框架：上传 / 录音 → Whisper 转录 → 显示文本

用法（项目根目录 F:\\condaEnv\\multimodel-emotion 下，先 conda activate emotion_ai）：
    python src\\app_day15.py                  # 启动界面，浏览器打开 http://127.0.0.1:7860
    python src\\app_day15.py --model base     # 换小一点的 Whisper（默认 small，和 Day 3 一致）
    python src\\app_day15.py --selftest 20    # 不开界面：dev 前 20 条走同一个回调函数，和 MELD 标注比 WER
    python src\\app_day15.py --share          # 额外生成 72 小时临时公网链接（Day 17 部署失败时的备选）

说明：
- 界面文字用英文（Demo 之后要放进 README / 发给招生老师看），代码注释用中文。
- 音频统一读成 16kHz 单声道 float32（和训练时一样），超过 30 秒只取前 30 秒。
- Whisper 幻觉过滤（Day 3 发现的问题）：
    ① transcribe 时 condition_on_previous_text=False（上一段的输出不再当提示，避免越写越多）；
    ② 用 Whisper 自带的静音判断：no_speech_prob > 0.6 且 avg_logprob < -1 的段落丢掉；
    ③ 起始时间超过音频真实长度的段落丢掉（短音频被补零到 30 秒，幻觉常出在补零那一段）；
    ④ 含中日韩字符的段落丢掉（已经指定 language="en"，出现这些字就是幻觉）；
    ⑤ 典型的 YouTube 结尾句（"thank you for watching" 等）丢掉；
    ⑥ 剩下的文字如果每秒超过 6 个词，只提示可能有幻觉，不删。
- 右边的"Emotion"框 Day 16 接入情绪模型，今天先空着。
"""
import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

import gradio as gr
import jiwer
import numpy as np
import pandas as pd
import torch
import whisper
from whisper.normalizers import EnglishTextNormalizer

from whisper_wer import DATA, MODEL_DIR, OUT_DIR, ROOT, SR, fix_text, load_wav, normalize

# Gradio 显示音频前会用 ffprobe 检查"浏览器能不能播放"，但它只检查了 ffmpeg 在不在 PATH 上。
# 本机的 ffmpeg.exe 是 Day 2 从 imageio-ffmpeg 复制来的，没有 ffprobe.exe → 启动时直接报错。
# 示例都是 16-bit PCM wav，浏览器一定能播 → 没有 ffprobe 时跳过这个检查。
# （装了 ffprobe 或部署到 HF Spaces 时这段不起作用，走 Gradio 原来的逻辑。）
HAVE_FFPROBE = shutil.which("ffprobe") is not None
if not HAVE_FFPROBE:
    import gradio.processing_utils as _gr_pu
    _gr_pu.audio_is_playable = lambda path: True

MAX_SEC = 30.0          # demo 最多处理 30 秒（Whisper 一个窗口就是 30 秒）
MIN_SEC = 0.1           # 短于 0.1 秒当成空输入
NO_SPEECH_P = 0.6       # 和 Whisper 默认阈值相同
LOGPROB_MIN = -1.0
MAX_WPS = 6.0           # 正常语速约 2–4 词/秒，超过 6 很可疑
HALLU_PHRASES = (       # 只匹配完整的 YouTube 腔句子，不匹配单独的 "Thank you."（MELD 里很常见）
    "thank you for watching", "thanks for watching", "please subscribe", "like and subscribe",
    "see you in the next video", "i hope you enjoyed this", "subtitles by", "amara.org",
)
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

device = "cuda" if torch.cuda.is_available() else "cpu"
asr = None              # main() 里加载一次，所有请求共用
norm = EnglishTextNormalizer()


# ---------- 1. 读音频 ----------
def read_audio(path):
    """任意格式 → 16kHz 单声道 float32。
    wav/flac/ogg/mp3 走 soundfile（快）；读不了（m4a、webm 等）就直接调 ffmpeg 解码成 16k 单声道 float32。
    只用 ffmpeg、不用 ffprobe（本机没有 ffprobe）。librosa 在新版本里已经不再自动用 ffmpeg 读，所以不靠它。"""
    try:
        return load_wav(path)
    except Exception:
        pass
    if shutil.which("ffmpeg") is None:
        raise gr.Error("This audio format needs ffmpeg. Please upload a .wav / .flac / .mp3 file.")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]          # 输出原始 float32 到 stdout
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise gr.Error("Could not decode this audio file: " + p.stderr.decode(errors="ignore")[-200:])
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


# ---------- 2. 转录 + 过滤 ----------
def segment_problem(seg, dur):
    """返回这个段落该丢掉的原因；没问题返回空字符串"""
    text = seg["text"].strip()
    if not text:
        return "empty"
    if seg["no_speech_prob"] > NO_SPEECH_P and seg["avg_logprob"] < LOGPROB_MIN:
        return "no speech"
    if seg["start"] > dur + 0.5:
        return "after end of audio"
    if CJK.search(text):
        return "non-English characters"
    low = text.lower()
    if any(p in low for p in HALLU_PHRASES):
        return "known hallucination phrase"
    return ""


def transcribe(y):
    """y：16kHz float32。返回 (保留的文本, 段落列表, 用时秒)"""
    t0 = time.time()
    out = asr.transcribe(y, language="en", fp16=(device == "cuda"),
                         condition_on_previous_text=False,
                         no_speech_threshold=NO_SPEECH_P, logprob_threshold=LOGPROB_MIN)
    dur = len(y) / SR
    segs = []
    for s in out["segments"]:
        segs.append(dict(start=s["start"], end=s["end"], text=s["text"].strip(),
                         no_speech_prob=s["no_speech_prob"], avg_logprob=s["avg_logprob"],
                         dropped=segment_problem(s, dur)))
    text = " ".join(s["text"] for s in segs if not s["dropped"]).strip()
    return text, segs, time.time() - t0


# ---------- 3. 界面回调 ----------
def run(audio_path, ref="", gold=""):
    """按钮回调：返回 (转录文本, 说明 Markdown)。selftest 也调用这个函数，测的就是界面走的那条路。"""
    if not audio_path:
        return "", "⚠️ Please upload or record an audio clip first."
    y = read_audio(audio_path)
    dur = len(y) / SR
    if dur < MIN_SEC or float(np.abs(y).max(initial=0.0)) < 1e-4:
        return "", f"⚠️ The clip is empty or silent ({dur:.2f} s)."
    notes = []
    if dur > MAX_SEC:
        y = y[: int(MAX_SEC * SR)]
        notes.append(f"Clip is {dur:.1f} s; only the first {MAX_SEC:.0f} s were used.")
        dur = MAX_SEC

    text, segs, secs = transcribe(y)
    n_words = len(text.split())
    if n_words > 8 and n_words / dur > MAX_WPS:
        notes.append(f"{n_words / dur:.1f} words/s is unusually fast — the transcript may contain hallucinated text.")
    if not text:
        notes.append("No speech was recognised.")

    lines = [f"**Audio** {dur:.2f} s · **ASR** whisper-{asr_name} on {device} · {secs:.2f} s"]
    if ref:
        r, h = normalize(norm, fix_text(ref)), normalize(norm, text)
        if r:
            lines.append(f"**MELD reference** {fix_text(ref)}  \n**WER** {jiwer.wer(r, h):.3f}"
                         + (f" · **gold emotion** {gold}" if gold else ""))
    if notes:
        lines.append("\n".join("⚠️ " + n for n in notes))
    if segs:
        lines.append("| start | end | no-speech p | avg logprob | text | kept? |\n|---|---|---|---|---|---|")
        for s in segs:
            kept = "✓" if not s["dropped"] else "✗ " + s["dropped"]
            safe = s["text"].replace("|", "/")
            lines[-1] += (f"\n| {s['start']:.1f} | {s['end']:.1f} | {s['no_speech_prob']:.2f} "
                          f"| {s['avg_logprob']:.2f} | {safe} | {kept} |")
    return text, "\n\n".join(lines)


# ---------- 4. 示例音频（MELD dev，每类情绪一条） ----------
def pick_examples(per_class=1):
    csv = DATA / "annotations" / "dev_sent_emo.csv"
    if not csv.exists():
        return []
    df = pd.read_csv(csv, encoding="utf-8")
    df["text"] = df["Utterance"].map(fix_text)
    df["n"] = df["text"].str.split().str.len()
    rows = []
    for emo in EMOTIONS:
        g = df[(df.Emotion == emo) & df.n.between(5, 14)]   # 5–14 个词：不太短、不太长
        for r in g.itertuples():
            wav = DATA / "audio" / "dev" / f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.wav"
            if wav.exists():
                rows.append([str(wav), r.text, emo])
                if sum(x[2] == emo for x in rows) >= per_class:
                    break
    return rows


# ---------- 5. 搭界面 ----------
def build_demo():
    examples = pick_examples()
    with gr.Blocks(title="Multimodal Emotion Recognition") as demo:
        gr.Markdown(
            "# 🎙️ Multimodal Emotion Recognition\n"
            "Upload or record a short English utterance. "
            "Step 1 (today): speech → text with Whisper. "
            "Step 2 (next): text + voice → emotion with a RoBERTa + wav2vec 2.0 late-fusion model trained on MELD."
        )
        with gr.Row():
            with gr.Column(scale=1):                                   # 左：输入区
                audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", format=None,
                                    label="Input audio")  # format=None：不让 Gradio 转格式（要 ffprobe），交给 read_audio
                with gr.Row():
                    btn = gr.Button("Transcribe", variant="primary")
                    clear = gr.Button("Clear")
                with gr.Accordion("Reference (filled in by the MELD examples)", open=False):
                    ref_box = gr.Textbox(label="MELD transcript", lines=2)
                    gold_box = gr.Textbox(label="MELD emotion label")
                if examples:
                    gr.Examples(examples=examples, inputs=[audio_in, ref_box, gold_box],
                                label="MELD dev examples (one per emotion)")
            with gr.Column(scale=1):                                   # 右：结果区
                text_out = gr.Textbox(label="Transcript (Whisper)", lines=3, interactive=False)
                emo_out = gr.Label(label="Emotion (coming on Day 16)", num_top_classes=7)
                info_out = gr.Markdown()

        btn.click(run, inputs=[audio_in, ref_box, gold_box], outputs=[text_out, info_out])
        # 用户自己上传 / 录音时，清掉示例留下的参考文本，免得 WER 对错句子
        audio_in.upload(lambda: ("", ""), None, [ref_box, gold_box])
        audio_in.stop_recording(lambda: ("", ""), None, [ref_box, gold_box])
        clear.click(lambda: (None, "", "", "", None, ""), None,
                    [audio_in, ref_box, gold_box, text_out, emo_out, info_out])
    return demo, examples


# ---------- 6. 不开界面的自测 ----------
def selftest(n):
    csv = DATA / "annotations" / "dev_sent_emo.csv"
    df = pd.read_csv(csv, encoding="utf-8")
    rows = []
    for r in df.itertuples():
        wav = DATA / "audio" / "dev" / f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.wav"
        if not wav.exists():
            continue
        t0 = time.time()
        text, info = run(str(wav), r.Utterance, r.Emotion)
        ref_n, hyp_n = normalize(norm, fix_text(r.Utterance)), normalize(norm, text)
        rows.append(dict(file=wav.name, emotion=r.Emotion, ref=fix_text(r.Utterance), hyp=text,
                         ref_norm=ref_n, hyp_norm=hyp_n, dropped=info.count("✗"),
                         warned=("⚠️" in info), secs=round(time.time() - t0, 2)))
        if len(rows) >= n:
            break
    res = pd.DataFrame(rows)
    ok = res[res.ref_norm.str.len() > 0]
    print(f"自测 {len(res)} 条（dev 前 {n} 条有 wav 的样本），平均每条 {res.secs.mean():.2f}s")
    print(f"corpus WER = {jiwer.wer(ok.ref_norm.tolist(), ok.hyp_norm.tolist()):.3f}")
    print(f"有段落被过滤的 {int((res.dropped > 0).sum())} 条，带警告的 {int(res.warned.sum())} 条，转录为空的 {int((res.hyp == '').sum())} 条")
    for r in res[(res.dropped > 0) | res.warned].itertuples():
        hyp = r.hyp if len(r.hyp) <= 120 else r.hyp[:120] + " …"
        print(f"  {r.file}  MELD: {r.ref}\n  {' ' * len(r.file)}  ASR : {hyp}")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "day15_selftest.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")


def main():
    global asr, asr_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="Whisper 模型：tiny / base / small / medium")
    ap.add_argument("--selftest", type=int, default=0, help="N > 0：不开界面，dev 前 N 条跑一遍回调函数")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成临时公网链接")
    ap.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    a = ap.parse_args()

    asr_name = a.model
    print(f"加载 whisper-{a.model} 到 {device}（缓存 {MODEL_DIR}）...")
    t0 = time.time()
    asr = whisper.load_model(a.model, device=device, download_root=MODEL_DIR)
    print(f"  完成，{time.time() - t0:.1f}s")

    if a.selftest:
        selftest(a.selftest)
        return

    demo, examples = build_demo()
    print(f"示例音频 {len(examples)} 条" + ("" if examples else "（没找到 data\\MELD，界面照常可用）"))
    # allowed_paths：示例 wav 在 data\MELD\audio 下，Gradio 5+ 默认只允许读工作目录和临时目录
    demo.queue().launch(server_name="127.0.0.1", server_port=a.port, share=a.share,
                        inbrowser=not a.no_browser, show_error=True, allowed_paths=[str(DATA / "audio")])


asr_name = "small"

if __name__ == "__main__":
    main()
