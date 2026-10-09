"""
Day 16 — 完整 Gradio Demo：上传 / 录音 → Whisper 转录 → 情绪预测（text + voice late fusion）→ 概率条 + 波形 / 语谱图

用法（项目根目录 F:\\condaEnv\\multimodel-emotion 下，先 conda activate emotion_ai）：
    python src\\app_day16.py --selftest 20           # 不开界面：dev 前 20 条走同一条流程，核对"在线提的特征 / 预测"和训练时一致
    python src\\app_day16.py --model turbo           # 启动界面（本机用 turbo；不加 --model 就是 small，HF Spaces 用 small）
    python src\\app_day16.py --asr-eval dev test     # 可选分析：Whisper 转录代替 MELD 标注文本，情绪模型掉多少（不用 GPU 跑 Whisper，读 Day 3 的转录）
    python src\\app_day16.py --share                 # 72 小时临时公网链接（Day 17 部署失败时的备选）
    python src\\app_day16.py --no-compare            # 不加载 text-only 模型（省 500MB 显存 / 内存，界面只显示 2 个模型）
    python src\\app_day16.py --host 0.0.0.0          # Day 17：监听所有网卡（局域网 / 容器里用；默认 127.0.0.1 只有本机能访问）

Day 17 小改（不影响 Day 16 的任何结果）：
  - build_demo(examples=None, examples_label=None)：可以传入自己的示例（HF Space 用自己录的音频，不放 MELD 片段）；
    不传就和 Day 16 一样自动挑 MELD 示例
  - main 加 --host（默认读环境变量 GRADIO_SERVER_NAME，没设就是 127.0.0.1，和 Day 16 一样）
  - HF Space 的入口是 space\\app.py，它直接 import 这里的 load_all / build_demo / run，不走 main()

流程（每一步都复现训练时的做法，否则模型看到的输入和训练时不一样）：
  1. read_audio（Day 15）→ 16kHz 单声道 float32；超过 30 秒只取前 30 秒
  2. Whisper 先检测语言（detect_language，看前 30 秒）：
       英文 → 和 Day 15 完全一样的转录 + 5 条过滤规则
       非英文（Day 15 定的方案 ②）→ 用检测到的语言转录（"中日韩字符"那条规则不再适用），情绪只用 voice-only 模型，
       界面标注 "not validated"（模型只在英文《老友记》上训练过）
  3. 音频特征 = Day 9 的做法：wav2vec2-base，fp32，逐条、不补零、不传 mask，前 20 秒，不足 400 采样点补零，
     13 层 hidden states 各自对时间求平均 → (13, 768)，再转成 float16（训练时特征就是 float16 存的）
  4. 文本 = Whisper 转录 → fix_text → RoBERTa tokenizer（max_length 128，和训练一样）
  5. 三个模型（都是 seed 42、按 dev 选的最佳 epoch，按 .json 的 config / model_kwargs 建，和 test_day14 一样）：
       text + voice  results\\fusion_best.pt            ← 主模型
       text only     results\\fusion_day13_text.pt      ← 对照（--no-compare 不加载）
       voice only    results\\fusion_day13_audio_e20.pt ← 对照；转录为空或非英文时它是主模型
     概率 = softmax(logits / T)，T 在 dev 上拟合（Day 14 的 temperature scaling；第一次运行时对三个模型各拟合一次，
     存 results\\demo_temperatures.json，之后直接读——Day 17 部署到 Spaces 时带上这个文件就行）
  6. 界面：结论（情绪 + 置信度 + 用的哪个模型）+ gr.Label 概率条 + 三个模型对照图 + 波形 / 语谱图 + 细节表
--selftest 额外核对（Day 16 的关键检查）：
  - 在线提的 wav2vec2 特征 vs Day 9 存的 .npz 同一条：最大绝对差应 ≈ 0（float16 舍入级别）
  - 用 MELD 标注文本 + 在线特征预测 vs Day 12 存的 day12_dev_probs_s42.npy：argmax 应 100% 一致
  - 再用 Whisper 转录走完整 demo 流程：看 ASR 文本代替标注文本后准确率变化、语言检测有没有误判
输出：
  results\\demo_temperatures.json   三个模型的 T（dev 上拟合）
  results\\day16_selftest.csv       （--selftest）逐条结果
  results\\day16_asr_eval.json / day16_asr_eval.csv   （--asr-eval）
"""
import argparse
import json
import os
import time

# 必须在 import transformers 之前设置
os.environ.setdefault("HF_HOME", r"F:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gradio as gr
import jiwer
import numpy as np
import pandas as pd
import torch
import whisper
from matplotlib.figure import Figure
from transformers import AutoFeatureExtractor, AutoModel, RobertaTokenizer
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import LANGUAGES

import app_day15 as d15  # import 时顺带打上 Day 15 的 ffprobe 补丁
from app_day15 import LOGPROB_MIN, MAX_SEC, MAX_WPS, MIN_SEC, NO_SPEECH_P, read_audio, segment_problem
from audio_day8 import embed
from fusion_day10 import AUDIO_MODEL, LateFusionClassifier
from roberta_day4 import LABELS, MODEL_NAME
from whisper_wer import DATA, MODEL_DIR, OUT_DIR, SR, fix_text, normalize

assert LABELS == d15.EMOTIONS, "label 顺序和 Day 15 不一致"
K = len(LABELS)
FEAT_MAX_SEC = 20.0     # 和 Day 9 提特征时的截断一致
MIN_SAMPLES = 400       # wav2vec2 第一层卷积的窗口（25ms）
MAX_LEN = 128           # RoBERTa max_length，和训练一致
TIE_MARGIN = 0.05       # selftest：存的概率 top1 − top2 小于它算"近似打平"（bf16 舍入就可能翻转）
TONE_MIN = 0.30         # voice-only 最高类概率 ≥ 0.30 且和最终结论不同 → 提示"语气和用词不一致"
EN_MIN = 0.4            # Whisper 判断的英文概率 ≥ 0.4 就当英文（短句的语言检测不稳，宁可偏向英文）
EMOJI = dict(anger="😠", disgust="🤢", fear="😨", joy="😄", neutral="😐", sadness="😢", surprise="😲")

# key → (权重文件名, 界面名字, Day 12/13 存的 dev 概率（拟合 T 用）)
SPEC = {
    "mm": ("fusion_best", "text + voice", "day12_dev_probs_s42.npy"),
    "text": ("fusion_day13_text", "text only", "day13_text_dev_probs_s42.npy"),
    "audio": ("fusion_day13_audio_e20", "voice only", "day13_audio_e20_dev_probs_s42.npy"),
}
# 图里的颜色：参考配色的前三个分类色（三个之间色盲也分得开），文字一律用中性灰，不用系列色
COLOR = dict(mm="#2a78d6", text="#eb6834", audio="#1baf7a")
# 透明背景 + 中性灰文字：Gradio 浅色 / 深色主题下都看得清（Day 16 本机是深色主题，白底图很突兀）
INK, INK2, GRID = "#8b8a86", "#8b8a86", "#8b8a8640"
TEMP_FILE = OUT_DIR / "demo_temperatures.json"
MELD_EXAMPLES_LABEL = "MELD examples (one per emotion from dev + 2 test clips where tone and words disagree)"

device = "cuda" if torch.cuda.is_available() else "cpu"
amp = device == "cuda"   # 和训练 / 评估时一样：GPU 上 bf16 autocast
S = {}                   # 全局：asr / asr_name / fe / w2v / tok / models / meta / T（main 里加载一次，所有请求共用）
norm = EnglishTextNormalizer()


# ================================================================ 1. 加载
def load_fusion(stem):
    """按 .json 里的 config / model_kwargs 建模型再装权重（和 test_day14.build_fusion 一样，不手写参数）。"""
    js, pt = OUT_DIR / f"{stem}.json", OUT_DIR / f"{stem}.pt"
    if not (js.exists() and pt.exists()):
        return None, None
    meta = json.loads(js.read_text(encoding="utf-8"))
    model = LateFusionClassifier(tuple(meta["config"]["modalities"]), **meta["model_kwargs"])
    model.load_state_dict(torch.load(pt, map_location="cpu"))
    return model.to(device).eval(), meta


def apply_t(probs, T):
    """softmax(log p / T)：和 test_day14.apply_t 一样。log p 和 logits 只差每行一个常数，所以拿存好的概率就能拟合 T。"""
    z = np.log(probs + 1e-12) / T
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(probs, gold):
    """和 test_day14.fit_temperature 一样：0.25–8 之间对数均匀 701 个点，取 dev NLL 最小的 T。"""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 701))
    nll = [-np.log(apply_t(probs, t)[np.arange(len(gold)), gold] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def load_temperatures(keys):
    """读 demo_temperatures.json；缺哪个模型的 T 就用 Day 12/13 存的 dev 概率现拟合一次，再写回文件。"""
    temps = json.loads(TEMP_FILE.read_text(encoding="utf-8")) if TEMP_FILE.exists() else {}
    missing = [k for k in keys if k not in temps]
    if missing:
        gold = None
        try:
            from fusion_day10 import build_table   # 要 data\MELD 的 CSV 和 Day 9 的 dev .npz，只有本机有
            gold = build_table("dev", verbose=False)[0]["label"].to_numpy()
        except Exception as e:
            print(f"  ⚠️ 拿不到 dev 标签（{e}），{missing} 的 T 先用 1.0")
        for k in missing:
            p = OUT_DIR / SPEC[k][2]
            if gold is not None and p.exists():
                probs = np.load(p)
                assert len(probs) == len(gold), f"{p.name} 有 {len(probs)} 行，dev 有 {len(gold)} 条"
                T = fit_temperature(probs, gold)
                acc = float((probs.argmax(1) == gold).mean())
                temps[k] = dict(T=round(T, 3), fitted_on=f"dev {len(gold)} utterances ({p.name})", dev_acc=round(acc, 4))
                print(f"  {SPEC[k][1]:12s} 在 dev 上拟合 T = {T:.3f}（dev acc {acc:.4f}，用来核对概率文件没拿错）")
            else:
                print(f"  ⚠️ 找不到 {p.name}，{SPEC[k][1]} 的 T 先用 1.0（概率会偏自信）")
        if gold is not None:
            ok = {k: v for k, v in temps.items() if isinstance(v, dict)}
            TEMP_FILE.write_text(json.dumps(ok, indent=2), encoding="utf-8")
            print(f"  已存 {TEMP_FILE}")
    out = {k: float(temps[k]["T"]) if k in temps else 1.0 for k in keys}
    ref = OUT_DIR / "fusion_best_temperature.json"
    if "mm" in out and ref.exists():
        t14 = json.loads(ref.read_text(encoding="utf-8"))["temperature"]
        print(f"  text + voice 的 T = {out['mm']:.3f}（Day 14 记录 {t14:.3f}{'，一致 ✅' if abs(out['mm'] - t14) < 0.01 else '，⚠️ 不一致'}）")
    return out


def load_all(whisper_name, compare=True, need_asr=True):
    t0 = time.time()
    if need_asr:
        print(f"加载 whisper-{whisper_name} 到 {device}（缓存 {MODEL_DIR}）...")
        S["asr"] = whisper.load_model(whisper_name, device=device, download_root=MODEL_DIR)
        S["asr_name"] = os.path.basename(str(whisper_name)).replace(".pt", "")
        d15.asr, d15.asr_name = S["asr"], S["asr_name"]
        print(f"加载 {AUDIO_MODEL}（音频特征，fp32）...")
        S["fe"] = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        S["w2v"] = AutoModel.from_pretrained(AUDIO_MODEL).to(device).eval()
    S["tok"] = RobertaTokenizer.from_pretrained(MODEL_NAME)
    S["models"], S["meta"] = {}, {}
    for k, (stem, name, _) in SPEC.items():
        if k == "text" and not compare:
            continue
        m, meta = load_fusion(stem)
        if m is None:
            if k == "mm":
                raise SystemExit(f"❌ 找不到 results\\{stem}.pt / .json（主模型），先跑 Day 12")
            print(f"  （{stem}.pt 不在，界面里不显示 {name}）")
            continue
        S["models"][k], S["meta"][k] = m, meta
        print(f"  {name:12s} ← {stem}.pt（seed {meta['config']['seed']}，epoch {meta['best_epoch']}，dev wF1 {meta['dev_wf1']}）")
    S["T"] = load_temperatures(list(S["models"]))
    print(f"全部加载完成，{time.time() - t0:.1f}s")
    if device == "cuda":
        print(f"显存占用 {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ================================================================ 2. 语言检测 + 转录
def detect_language(y):
    """Whisper 自带的语言检测：只看前 30 秒的 log-mel。返回 {语言代码: 概率}。"""
    asr = S["asr"]
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(y), n_mels=asr.dims.n_mels).to(asr.device)
    _, probs = asr.detect_language(mel)
    return probs


def transcribe16(y, lang_mode="Auto-detect"):
    """→ dict(text, segs, english, lang, p_en, top, top_p, secs)。英文时和 Day 15 的 transcribe 完全一样。"""
    t0 = time.time()
    probs = detect_language(y)
    top = max(probs, key=probs.get)
    p_en = float(probs.get("en", 0.0))
    english = lang_mode == "English" or top == "en" or p_en >= EN_MIN
    lang = "en" if english else top
    out = S["asr"].transcribe(y, language=lang, fp16=(device == "cuda"), condition_on_previous_text=False,
                              no_speech_threshold=NO_SPEECH_P, logprob_threshold=LOGPROB_MIN)
    dur = len(y) / SR
    segs = []
    for s in out["segments"]:
        why = segment_problem(s, dur)
        if not english and why == "non-English characters":   # 非英文时出现中日韩字符是正常的
            why = ""
        segs.append(dict(start=s["start"], end=s["end"], text=s["text"].strip(), no_speech_prob=s["no_speech_prob"],
                         avg_logprob=s["avg_logprob"], dropped=why))
    text = " ".join(s["text"] for s in segs if not s["dropped"]).strip()
    return dict(text=text, segs=segs, english=english, lang=lang, p_en=p_en, top=top, top_p=float(probs[top]),
                secs=time.time() - t0)


# ================================================================ 3. 特征 + 情绪预测
@torch.no_grad()
def audio_features(y):
    """Day 9 的做法：前 20 秒、不足 400 采样点补零、fp32 逐条 → 13 层 mean pooling (13, 768) → float16。"""
    y = y[: int(FEAT_MAX_SEC * SR)]
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)))
    e = embed(S["w2v"], S["fe"], [y], device)[0]          # (13, 768) float32
    return e.to(torch.float16).cpu()                       # 训练时特征是 float16 存的，这里也舍入一次


@torch.no_grad()
def predict(key, text, feats):
    """一个模型、一条样本 → dict(raw=未校准概率, cal=softmax(logits / T))，都是长度 7 的 numpy。"""
    m = S["models"][key]
    kw = {}
    if "text" in m.modalities:
        enc = S["tok"]([fix_text(text)], truncation=True, max_length=MAX_LEN, return_tensors="pt")
        kw.update(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
    if "audio" in m.modalities:
        kw["audio"] = feats[None].to(device)               # (1, 13, 768)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, _ = m(**kw)
    z = logits.float()[0].cpu()
    return dict(raw=z.softmax(-1).numpy(), cal=(z / S["T"][key]).softmax(-1).numpy())


def analyse(y, lang_mode="Auto-detect", text_override=None):
    """完整流程（界面和 selftest 共用）。text_override：selftest 用 MELD 标注文本代替 Whisper 转录做一致性核对。"""
    r = dict(dur=len(y) / SR)
    if text_override is None:
        r.update(transcribe16(y, lang_mode))
    else:
        r.update(text=text_override, segs=[], english=True, lang="en", p_en=1.0, top="en", top_p=1.0, secs=0.0)
    t0 = time.time()
    feats = audio_features(y)
    r["feats"], r["w2v_secs"] = feats, time.time() - t0

    t0 = time.time()
    has_text = bool(r["text"])
    preds = {}
    for k in S["models"]:
        if "text" in S["models"][k].modalities and not (has_text and r["english"]):
            continue                                       # 没有可用的英文文本 → 带文本的模型不预测
        preds[k] = predict(k, r["text"], feats)
    r["preds"], r["clf_secs"] = preds, time.time() - t0
    if r["english"] and has_text:
        r["route"] = "mm"
    elif "audio" in preds:
        r["route"] = "audio"
    else:
        r["route"] = None
    if not r["english"]:
        r["route_note"] = (f"Non-English input detected ({LANGUAGES.get(r['lang'], r['lang']).title()}, "
                           f"p={r['top_p']:.2f}): emotion from voice only (model trained on English, not validated).")
    elif not has_text:
        r["route_note"] = "No speech was recognised: emotion from voice only."
    else:
        r["route_note"] = ""
    return r


# ================================================================ 4. 图
def prob_figure(preds, route):
    """三个模型的 7 类概率（已校准），横向分组条形图。主模型那一组画在最上面、数字只标每个模型的最高类。"""
    keys = [k for k in ("mm", "text", "audio") if k in preds]
    fig = Figure(figsize=(6.4, 4.2), dpi=110)
    fig.patch.set_alpha(0)
    ax = fig.subplots()
    ax.set_facecolor("none")
    n = len(keys)
    h = 0.8 / max(n, 1)
    ys = np.arange(K)[::-1]                                # anger 在最上面
    for j, k in enumerate(keys):
        p = preds[k]["cal"]
        off = (j - (n - 1) / 2) * h
        ax.barh(ys - off, p, height=h * 0.86, color=COLOR[k], alpha=1.0 if k == route else 0.8,
                label=SPEC[k][1] + ("  (used)" if k == route else ""))
        i = int(p.argmax())
        ax.text(p[i] + 0.01, ys[i] - off, f"{p[i]:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, LABELS, fontsize=9, color=INK)   # matplotlib 默认字体没有 emoji，标签只写英文
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Emotion probabilities by model (temperature-scaled)", fontsize=10, color=INK, loc="left")
    if n:
        ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    return fig


def audio_figure(y):
    """上：波形；下：线性频率语谱图（0–4 kHz，n_fft 512 / hop 160，和 Day 9 的语谱图同设置）。只用 numpy，不依赖 librosa。"""
    dur = len(y) / SR
    fig = Figure(figsize=(6.4, 3.6), dpi=110)
    fig.patch.set_alpha(0)
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 2]))
    ax1.set_facecolor("none")
    step = max(1, len(y) // 4000)                          # 画图只取约 4000 个点
    t = np.arange(len(y))[::step] / SR
    ax1.plot(t, y[::step], lw=0.6, color=COLOR["mm"])
    ax1.set_ylabel("amplitude", fontsize=8, color=INK2)
    ax1.tick_params(labelsize=7, colors=INK2)
    n_fft, hop = 512, 160
    yy = np.pad(y, (0, max(0, n_fft - len(y))))
    frames = 1 + (len(yy) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(yy[idx] * np.hanning(n_fft), axis=1)).T   # (257, frames)
    db = 20 * np.log10(spec / (spec.max() + 1e-12) + 1e-6)
    fmax = 4000
    nb = int(fmax / (SR / 2) * (n_fft // 2)) + 1
    ax2.imshow(np.clip(db[:nb], -80, 0), origin="lower", aspect="auto", cmap="magma",
               extent=[0, dur, 0, fmax])
    ax2.set_ylabel("Hz", fontsize=8, color=INK2)
    ax2.set_xlabel("time (s)", fontsize=8, color=INK2)
    ax2.tick_params(labelsize=7, colors=INK2)
    for a in (ax1, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
    ax1.set_title(f"Waveform and spectrogram ({dur:.2f} s)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig


# ================================================================ 5. 界面回调
def run(audio_path, lang_mode="Auto-detect", ref="", gold=""):
    """按钮回调 → (转录, 结论 Markdown, gr.Label 的 {类别: 概率}, 概率图, 波形图, 细节 Markdown)。"""
    empty = ("", "", None, None, None)
    if not audio_path:
        return ("", "⚠️ Please upload or record an audio clip first.") + empty[2:] + ("",)
    y = read_audio(audio_path)
    dur = len(y) / SR
    if dur < MIN_SEC or float(np.abs(y).max(initial=0.0)) < 1e-4:
        return ("", f"⚠️ The clip is empty or silent ({dur:.2f} s).") + empty[2:] + ("",)
    notes = []
    if dur > MAX_SEC:
        y = y[: int(MAX_SEC * SR)]
        notes.append(f"Clip is {dur:.1f} s; only the first {MAX_SEC:.0f} s were used (emotion features use the first {FEAT_MAX_SEC:.0f} s).")
        dur = MAX_SEC
    elif dur > FEAT_MAX_SEC:
        notes.append(f"Voice features use the first {FEAT_MAX_SEC:.0f} s (as in training).")

    r = analyse(y, lang_mode)
    text, route, preds = r["text"], r["route"], r["preds"]
    n_words = len(text.split())
    if n_words > 8 and n_words / dur > MAX_WPS:
        notes.append(f"{n_words / dur:.1f} words/s is unusually fast — the transcript may contain hallucinated text.")
    if r["route_note"]:
        notes.insert(0, r["route_note"])

    # ---- 结论 ----
    if route is None:
        verdict = "⚠️ No usable model for this input (voice-only model not found)."
        label = None
    else:
        p = preds[route]["cal"]
        i = int(p.argmax())
        verdict = (f"## {EMOJI[LABELS[i]]} {LABELS[i]} · {p[i]:.0%}\n"
                   f"Model used: **{SPEC[route][1]}**" + (" (late fusion of RoBERTa + wav2vec 2.0)" if route == "mm" else ""))
        if gold:
            verdict += f"  \nMELD gold label: **{gold}** {'✓' if gold == LABELS[i] else '✗'}"
        if r["route_note"]:
            verdict += f"\n\n⚠️ {r['route_note']}"
        # Day 16 本机测试：同一句 "I'm fine." 换不同语气，voice only 的分布会变，但 text + voice 始终 90%+ neutral
        # （late fusion 被文本主导，Day 12 打乱测试也是这样）。不改模型，只把两路的分歧如实显示出来。
        if route == "mm" and "audio" in preds:
            pv = preds["audio"]["cal"]
            iv = int(pv.argmax())
            if iv != i and pv[iv] >= TONE_MIN:
                verdict += (f"\n\n🔊 **Tone vs words:** the voice-only model hears **{LABELS[iv]}** ({pv[iv]:.0%}), "
                            f"but the fused prediction follows the words (**{LABELS[i]}**). "
                            "In this late-fusion model the text usually dominates.")
        label = {l: float(v) for l, v in zip(LABELS, p)}

    # ---- 细节 ----
    lines = []
    lang_name = LANGUAGES.get(r["top"], r["top"]).title()
    lines.append(f"**Audio** {dur:.2f} s · **ASR** whisper-{S['asr_name']} on {device} · "
                 f"**language** {lang_name} (p={r['top_p']:.2f}; English p={r['p_en']:.2f}"
                 + (", forced English" if lang_mode == "English" else "") + ")  \n"
                 f"**time** ASR {r['secs']:.2f} s · wav2vec 2.0 {r['w2v_secs']:.2f} s · classifiers {r['clf_secs']:.2f} s")
    if ref:
        rr, hh = normalize(norm, fix_text(ref)), normalize(norm, text)
        if rr:
            lines.append(f"**MELD reference** {fix_text(ref)}  \n**WER** {jiwer.wer(rr, hh):.3f}")
    if preds:
        tab = ["| model | input | prediction | confidence | p(" + (gold or "gold") + ") |" if gold else
               "| model | input | prediction | confidence |", "|---|---|---|---|" + ("---|" if gold else "")]
        for k in ("mm", "text", "audio"):
            if k not in preds:
                continue
            p = preds[k]["cal"]
            i = int(p.argmax())
            inp = dict(mm="transcript + voice", text="transcript", audio="voice")[k]
            row = f"| {SPEC[k][1]}{' ★' if k == route else ''} | {inp} | {EMOJI[LABELS[i]]} {LABELS[i]} | {p[i]:.0%} |"
            if gold:
                row += f" {p[LABELS.index(gold)]:.0%} |"
            tab.append(row)
        lines.append("\n".join(tab))
    if notes:
        lines.append("\n".join("⚠️ " + n for n in notes))
    if r["segs"]:
        seg = "| start | end | no-speech p | avg logprob | text | kept? |\n|---|---|---|---|---|---|"
        for s in r["segs"]:
            kept = "✓" if not s["dropped"] else "✗ " + s["dropped"]
            seg += (f"\n| {s['start']:.1f} | {s['end']:.1f} | {s['no_speech_prob']:.2f} | {s['avg_logprob']:.2f} "
                    f"| {s['text'].replace('|', '/')} | {kept} |")
        lines.append(seg)
    lines.append("<sub>Probabilities are temperature-scaled (T fitted on MELD dev: "
                 + ", ".join(f"{SPEC[k][1]} {S['T'][k]:.2f}" for k in S["models"])
                 + "). Models were trained on MELD (English dialogue from the TV series *Friends*), 7 classes; "
                 "text + voice test weighted F1 ≈ 0.64 (3 seeds). Treat predictions on other speakers / languages as a demo, not a measurement.</sub>")
    return text, verdict, label, prob_figure(preds, route), audio_figure(y), "\n\n".join(lines)


# ================================================================ 6. 示例音频
SHOWCASE = [("test", 252, 5), ("test", 116, 2)]   # Day 14 错例：文字平静、语气愤怒（text 判 neutral，voice-only 判 anger）


def pick_examples16():
    """每类情绪一条 dev 示例：优先选 Whisper-small 转录和标注完全一致（WER = 0）、5–14 个词、8 秒以内的句子
    （避开 Day 15 发现的“片段混进邻句”问题），没有 transcripts_dev_small.csv 就退回 Day 15 的选法；
    再加 2 条 test 上“文本把语气盖过去”的错例。按 CSV 顺序取第一条，不按模型对错挑。"""
    rows = []
    tr = OUT_DIR / "transcripts_dev_small.csv"
    if tr.exists():
        t = pd.read_csv(tr, keep_default_na=False)
        t["n"] = t["ref_norm"].str.split().str.len()
        ok = t[(t["wer"] == 0) & t["n"].between(5, 14) & (t["duration_s"] < 8)]
        for emo in LABELS:
            g = ok[ok["emotion"] == emo]
            for r in g.itertuples():
                wav = DATA / "audio" / "dev" / r.file
                if wav.exists():
                    rows.append([str(wav), "Auto-detect", r.ref, emo])
                    break
    if len(rows) < K:
        have = {x[3] for x in rows}
        rows += [[w, "Auto-detect", t_, e] for w, t_, e in d15.pick_examples() if e not in have]
    for split, d, u in SHOWCASE:
        csv = DATA / "annotations" / f"{split}_sent_emo.csv"
        wav = DATA / "audio" / split / f"dia{d}_utt{u}.wav"
        if csv.exists() and wav.exists():
            df = pd.read_csv(csv, encoding="utf-8")
            r = df[(df.Dialogue_ID == d) & (df.Utterance_ID == u)]
            if len(r):
                rows.append([str(wav), "Auto-detect", fix_text(r.Utterance.iloc[0]), r.Emotion.iloc[0]])
    return rows


def build_demo(examples=None, examples_label=None):
    """examples：[[音频路径, 语言, 参考文本, 情绪标签], ...]；None = 和 Day 16 一样自动挑 MELD 示例（Day 17 加的参数）。
    examples_label：示例区的标题；None = MELD 示例的标题。"""
    if examples is None:
        examples = pick_examples16()
    n_models = len(S["models"])
    with gr.Blocks(title="Multimodal Emotion Recognition") as demo:
        gr.Markdown(
            "# 🎙️ Multimodal Emotion Recognition\n"
            "Upload or record a short English utterance. Whisper transcribes it; a **late-fusion** model combines "
            "**what was said** (RoBERTa) with **how it was said** (wav2vec 2.0) to predict one of 7 emotions. "
            f"The chart compares {'all three models' if n_models == 3 else 'the loaded models'}: text + voice, "
            + ("text only, " if "text" in S["models"] else "") + "voice only. Trained on MELD (*Friends*)."
        )
        with gr.Row():
            with gr.Column(scale=1):                                   # 左：输入
                audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", format=None, label="Input audio")
                lang_in = gr.Radio(["Auto-detect", "English"], value="Auto-detect", label="Language",
                                   info="Non-English speech gets a voice-only prediction (not validated).")
                with gr.Row():
                    btn = gr.Button("Analyse", variant="primary")
                    clear = gr.Button("Clear")
                with gr.Accordion("Reference (filled in by the MELD examples)", open=False):
                    ref_box = gr.Textbox(label="MELD transcript", lines=2)
                    gold_box = gr.Textbox(label="MELD emotion label")
                if examples:
                    gr.Examples(examples=examples, inputs=[audio_in, lang_in, ref_box, gold_box],
                                label=examples_label or MELD_EXAMPLES_LABEL)
                wave_out = gr.Plot(label="Audio", show_label=False)
            with gr.Column(scale=1):                                   # 右：结果
                verdict_out = gr.Markdown()
                emo_out = gr.Label(label="Emotion probabilities", num_top_classes=K)
                text_out = gr.Textbox(label="Transcript (Whisper)", lines=2, interactive=False)
                prob_out = gr.Plot(label="Model comparison", show_label=False)
                with gr.Accordion("Details", open=False):
                    info_out = gr.Markdown()

        outs = [text_out, verdict_out, emo_out, prob_out, wave_out, info_out]
        btn.click(run, inputs=[audio_in, lang_in, ref_box, gold_box], outputs=outs)
        audio_in.upload(lambda: ("", ""), None, [ref_box, gold_box])          # 用户自己的音频：清掉示例的参考
        audio_in.stop_recording(lambda: ("", ""), None, [ref_box, gold_box])
        clear.click(lambda: (None, "", "", "", "", None, None, None, ""), None,
                    [audio_in, ref_box, gold_box, text_out, verdict_out, emo_out, prob_out, wave_out, info_out])
    return demo, examples


# ================================================================ 7. 自测：在线流程 vs 训练时存的特征 / 概率
def selftest(n, split="dev"):
    from fusion_day10 import build_table
    df, emb, _ = build_table(split, verbose=False)        # 行顺序 = Day 12 存 dev 概率时的顺序
    stored = {}
    for k in S["models"]:
        p = OUT_DIR / SPEC[k][2]
        if split == "dev" and p.exists():
            stored[k] = np.load(p)
    rows = []
    for i, r in df.iterrows():
        wav = DATA / "audio" / split / f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.wav"
        if not wav.exists():
            continue
        y = read_audio(wav)
        y = y[: int(MAX_SEC * SR)]
        t0 = time.time()
        a_ref = analyse(y, text_override=r.text)                 # MELD 标注文本 + 在线特征（应和训练时一致）
        a_asr = analyse(y)                                        # 真实 demo 流程：Whisper 转录
        secs = time.time() - t0
        f_npz = torch.from_numpy(emb[r.audio_idx].astype(np.float32))
        f_now = a_ref["feats"].float()
        cos = torch.nn.functional.cosine_similarity(f_now, f_npz, dim=-1).min().item()
        row = dict(file=wav.name, gold=LABELS[r.label], ref=r.text, hyp=a_asr["text"], lang=a_asr["top"],
                   p_en=round(a_asr["p_en"], 3), route=a_asr["route"], feat_maxdiff=float((f_now - f_npz).abs().max()),
                   feat_cos_min=round(cos, 6), secs=round(secs, 2), asr_secs=round(a_asr["secs"], 2),
                   w2v_secs=round(a_asr["w2v_secs"], 3))
        rn, hn = normalize(norm, r.text), normalize(norm, a_asr["text"])
        row["wer"] = round(jiwer.wer(rn, hn), 3) if rn else np.nan
        for k in S["models"]:
            if k in a_ref["preds"]:
                p = a_ref["preds"][k]["raw"]
                row[f"{k}_pred_ref"] = LABELS[int(p.argmax())]
                if k in stored:
                    row[f"{k}_pred_stored"] = LABELS[int(stored[k][i].argmax())]
                    row[f"{k}_maxdiff"] = float(np.abs(p - stored[k][i]).max())
                    top2 = np.sort(stored[k][i])[-2:]
                    row[f"{k}_margin"] = float(top2[1] - top2[0])   # 存的概率里第一名比第二名高多少
            if k in a_asr["preds"]:
                row[f"{k}_pred_asr"] = LABELS[int(a_asr["preds"][k]["cal"].argmax())]
        row["demo_pred"] = LABELS[int(a_asr["preds"][a_asr["route"]]["cal"].argmax())] if a_asr["route"] else ""
        rows.append(row)
        print(f"  {wav.name:16s} [{row['gold']:8s}] demo→{row['demo_pred']:8s}（{row['route']}）  "
              f"ref 文本 mm→{row.get('mm_pred_ref', '-'):8s} 存的→{row.get('mm_pred_stored', '-'):8s}  "
              f"特征差 {row['feat_maxdiff']:.4f}  {r.text[:40]!r} → {a_asr['text'][:40]!r}")
        if len(rows) >= n:
            break
    res = pd.DataFrame(rows)
    print(f"\n========== 自测汇总（{split} 前 {len(res)} 条）==========")
    print(f"  ① 在线 wav2vec2 特征 vs Day 9 .npz：最大绝对差 {res.feat_maxdiff.max():.4f}，13 层余弦最小 {res.feat_cos_min.min():.6f}"
          f"  {'✅ 和训练时一致' if res.feat_cos_min.min() > 0.999 else '⚠️ 不一致：检查模型 / 截断 / 采样率'}")
    checks = {"features": bool(res.feat_cos_min.min() > 0.999)}
    for k in S["models"]:
        if f"{k}_pred_stored" in res:
            # Day 16 本机结果：bf16 下"一条一条推理"和"64 条一批、补 padding 推理"的舍入不同，概率会差 ~0.01–0.05；
            # 第一名和第二名几乎打平（差 < 0.05）的句子，argmax 可能因此翻转，这不是流程错误。
            # 所以只要求"明确的"句子（存的概率里 top1 − top2 ≥ 0.05）100% 一致，近似打平的单独列出来。
            same = res[f"{k}_pred_ref"] == res[f"{k}_pred_stored"]
            clear = res[f"{k}_margin"] >= TIE_MARGIN
            agree, agree_clear = float(same.mean()), float(same[clear].mean()) if clear.any() else 1.0
            flips = res[~same]
            print(f"  ② {SPEC[k][1]:12s} 标注文本 + 在线特征 vs Day 12/13 存的 dev 概率：argmax 一致 {agree:.1%}"
                  f"（明确的 {int(clear.sum())} 条：{agree_clear:.1%}），概率最大差 {res[f'{k}_maxdiff'].max():.4f}"
                  f"  {'✅' if agree_clear == 1.0 else '⚠️'}"
                  + "".join(f"\n     近似打平翻转：{f.file} 存的 {f[k + '_pred_stored']} / 现在 {f[k + '_pred_ref']}（存的 top1−top2 = {f[k + '_margin']:.3f}）"
                            for _, f in flips.iterrows()))
            checks[f"agree_{k}"] = agree_clear == 1.0
    non_en = res[res.lang != "en"]
    print(f"  ③ 语言检测：{len(non_en)} 条被判成非英文（MELD 全是英文，应为 0）；English p 最小 {res.p_en.min():.2f}"
          + (f"；判错的：{', '.join(non_en.file + '(' + non_en.lang + ')')}" if len(non_en) else ""))
    print(f"     走 voice-only 的 {int((res.route == 'audio').sum())} 条（转录为空或非英文）")
    ok = res[res.wer.notna()]
    print(f"  ④ Whisper-{S['asr_name']} corpus WER {jiwer.wer([normalize(norm, x) for x in ok.ref], [normalize(norm, x) for x in ok.hyp]):.3f}")
    acc_ref = (res.get("mm_pred_ref") == res.gold).mean()
    acc_demo = (res.demo_pred == res.gold).mean()
    print(f"  ⑤ 准确率（只有 {len(res)} 条，看个大概）：标注文本 + 语音 {acc_ref:.3f} → 真实 demo（Whisper 文本）{acc_demo:.3f}")
    for k in ("text", "audio"):
        if f"{k}_pred_asr" in res:
            how = "（Whisper 文本）" if k == "text" else "（只用语音）"
            print(f"     {SPEC[k][1]:12s}{how}{(res[f'{k}_pred_asr'] == res.gold).mean():.3f}")
    print(f"  ⑥ 每条平均用时 {res.secs.mean():.2f}s（两遍流程）；其中 ASR {res.asr_secs.mean():.2f}s，wav2vec2 {res.w2v_secs.mean():.3f}s")
    out = OUT_DIR / "day16_selftest.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    print("\n自测核对：" + "  ".join(f"{k} {'✅' if v else '⚠️'}" for k, v in checks.items()))
    return checks


# ================================================================ 8. 可选分析：Whisper 文本代替标注文本
@torch.no_grad()
def batch_probs(key, texts, feats, bs=64):
    """批量版 predict（只用于 --asr-eval）：texts 为 list[str]，feats (N, 13, 768) float16 → (N, 7) 未校准概率。"""
    m = S["models"][key]
    out = []
    for s in range(0, len(feats), bs):
        kw = {}
        if "text" in m.modalities:
            enc = S["tok"]([fix_text(t) for t in texts[s:s + bs]], truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt")
            kw.update(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        if "audio" in m.modalities:
            kw["audio"] = torch.from_numpy(feats[s:s + bs]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = m(**kw)
        out.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def asr_eval(splits, n_boot=2000):
    """Day 3 的结论里留的“真实 ASR 消融”：同一批句子，文本分别用 MELD 标注 / Whisper-small 转录（Day 3 存的 transcripts_*），
    音频用 Day 9 的特征。模型、权重都不变（只报数，不再选任何东西）。
    demo 路由 = 转录为空时改用 voice-only（和界面一样）。"""
    from ablation_day13 import paired_boot_mean, scores
    from fusion_day10 import build_table
    report, rows_out = {}, []
    for split in splits:
        tp = OUT_DIR / f"transcripts_{split}_small.csv"
        if not tp.exists():
            print(f"  ⚠️ 没有 {tp.name}（python src\\whisper_wer.py --all --split {split}），跳过 {split}")
            continue
        df, emb, _ = build_table(split, verbose=False)
        tr = pd.read_csv(tp, keep_default_na=False)[["dialogue_id", "utterance_id", "hyp", "wer"]]
        df = df.merge(tr, left_on=["Dialogue_ID", "Utterance_ID"], right_on=["dialogue_id", "utterance_id"], how="left")
        assert df["hyp"].notna().all(), f"{split}: 有句子没有转录"
        feats = emb[df["audio_idx"].to_numpy()]
        gold = df["label"].to_numpy()
        gold_txt, asr_txt = df["text"].tolist(), df["hyp"].astype(str).tolist()
        empty = np.array([not t.strip() for t in asr_txt])
        P = {}
        for k in S["models"]:
            if "text" in S["models"][k].modalities:
                P[f"{k}_gold"] = batch_probs(k, gold_txt, feats)
                P[f"{k}_asr"] = batch_probs(k, asr_txt, feats)
            else:
                P[k] = batch_probs(k, gold_txt, feats)
        if "audio" in P:
            P["demo"] = np.where(empty[:, None], P["audio"], P["mm_asr"])
        sc = {k: scores(gold, v.argmax(1)) for k, v in P.items()}
        corpus_wer = jiwer.wer([normalize(norm, t) or "-" for t in gold_txt], [normalize(norm, t) for t in asr_txt])
        print(f"\n========== {split}（{len(gold)} 条；Whisper-small corpus WER {corpus_wer:.3f}；转录为空 {int(empty.sum())} 条）==========")
        print(f"  {'模型':28s} {'acc':>7} {'wF1':>7} {'mF1':>7}")
        names = dict(mm_gold="text + voice · MELD text", mm_asr="text + voice · Whisper text",
                     text_gold="text only · MELD text", text_asr="text only · Whisper text", audio="voice only",
                     demo="demo routing (Whisper; empty → voice)")
        for k in ("mm_gold", "text_gold", "mm_asr", "text_asr", "audio", "demo"):
            if k in sc:
                print(f"  {names[k]:28s} {sc[k]['acc']:>7.4f} {sc[k]['wf1']:>7.4f} {sc[k]['mf1']:>7.4f}")
        boots = {}
        for a, b, desc in [("mm_asr", "mm_gold", "text+voice: Whisper text − MELD text"),
                           ("text_asr", "text_gold", "text only: Whisper text − MELD text"),
                           ("mm_gold", "text_gold", "voice adds (MELD text)"),
                           ("mm_asr", "text_asr", "voice adds (Whisper text)")]:
            if a in P and b in P and n_boot:
                bt = paired_boot_mean(gold, [P[a]], [P[b]], n=n_boot)
                boots[desc] = bt
                print(f"  {desc:38s} Δ wF1 {bt['diff']:+.4f}  95% [{bt['lo']:+.4f}, {bt['hi']:+.4f}]")
        # 按 WER 分档：转录越差掉得越多吗
        w = df["wer"].replace("", np.nan).astype(float).to_numpy()
        bins = [(0, 0, "WER = 0"), (1e-9, 0.5, "0 < WER ≤ 0.5"), (0.5 + 1e-9, np.inf, "WER > 0.5")]
        print("  按 WER 分档的 acc（text + voice：MELD 文本 → Whisper 文本）：" + "  ".join(
            f"{lab} {(P['mm_gold'].argmax(1) == gold)[(w >= lo) & (w <= hi)].mean():.3f} → "
            f"{(P['mm_asr'].argmax(1) == gold)[(w >= lo) & (w <= hi)].mean():.3f}（{int(((w >= lo) & (w <= hi)).sum())} 条）"
            for lo, hi, lab in bins if ((w >= lo) & (w <= hi)).any()))
        report[split] = dict(n=len(gold), corpus_wer=round(corpus_wer, 4), n_empty=int(empty.sum()),
                             scores={k: {m_: round(v_, 4) for m_, v_ in s_.items() if m_ != "per_class"} for k, s_ in sc.items()},
                             per_class={k: [round(x, 4) for x in s_["per_class"]] for k, s_ in sc.items()},
                             bootstrap={k: {m_: round(float(v_), 4) for m_, v_ in v.items()} for k, v in boots.items()})
        for k, s_ in sc.items():
            rows_out.append(dict(split=split, model=names.get(k, k), **{m_: round(v_, 4) for m_, v_ in s_.items() if m_ != "per_class"}))
    if report:
        (OUT_DIR / "day16_asr_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        pd.DataFrame(rows_out).to_csv(OUT_DIR / "day16_asr_eval.csv", index=False)
        print(f"\n已保存 results\\day16_asr_eval.json / .csv")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "small"),
                    help="Whisper：tiny / base / small / medium / turbo（默认 small，或环境变量 WHISPER_MODEL）")
    ap.add_argument("--selftest", type=int, default=0, help="N > 0：不开界面，dev 前 N 条做一致性核对")
    ap.add_argument("--asr-eval", nargs="*", choices=["dev", "test"], help="Whisper 文本代替标注文本的消融（不开界面）")
    ap.add_argument("--no-compare", action="store_true", help="不加载 text-only 对照模型")
    ap.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                    help="监听地址（Day 17）：127.0.0.1 只有本机能访问；0.0.0.0 = 局域网 / 容器")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print(f"device={device}  bf16={amp}  HF_HOME={os.environ['HF_HOME']}")

    if a.asr_eval is not None:
        load_all(a.model, compare=True, need_asr=False)
        asr_eval(a.asr_eval or ["dev", "test"])
        return
    load_all(a.model, compare=not a.no_compare)
    if a.selftest:
        selftest(a.selftest)
        return
    demo, examples = build_demo()
    print(f"示例音频 {len(examples)} 条" + ("" if examples else "（没找到 data\\MELD，界面照常可用）"))
    demo.queue().launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=not a.no_browser,
                        show_error=True, allowed_paths=[str(DATA / "audio")])


if __name__ == "__main__":
    main()
