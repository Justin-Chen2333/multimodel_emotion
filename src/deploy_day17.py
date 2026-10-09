"""
Day 17 — 部署到 HuggingFace Spaces：上传权重 → 组装 Space 文件夹 → 本机 CPU 演练 → 推送 → 查状态 → 从外部调用一次

先做一次（PowerShell，conda activate emotion_ai）：
    1. 注册 https://huggingface.co/join ，邮箱验证后在 Settings → Access Tokens 建一个 **Write** 权限的 token
    2. python -c "from huggingface_hub import login; login()"     # 粘贴 token；存到本机，以后不用再输

然后按顺序（项目根目录 F:\\condaEnv\\multimodel-emotion 下）：
    python src\\deploy_day17.py check             # 检查：登录了没有、要上传的文件在不在、本机包版本
    python src\\deploy_day17.py upload-weights    # 3 个 .pt + .json + 模型说明 → HF model repo（约 1 GB，看网速）
    python src\\deploy_day17.py build             # 组装 build\\hf_space\\（Space 仓库里的全部文件）
    python src\\deploy_day17.py rehearse          # 本机演练：强制 CPU、从 model repo 下载权重、跑 3 条 dev 音频，看每条几秒
    python src\\deploy_day17.py rehearse --launch # 演练界面：http://127.0.0.1:7861（和 Space 上看到的一样）
    python src\\deploy_day17.py push              # 建 Space（免费 CPU basic）并上传 build\\hf_space\\（不含 .pt）
    python src\\deploy_day17.py status            # 看 Space 构建 / 运行状态（BUILDING → RUNNING 要几分钟）
    python src\\deploy_day17.py ping              # 用 gradio_client 从外部调用 Space 一次 = 别人也能用

常用参数：
    --model-repo NAME   权重仓库名（默认 meld-emotion-late-fusion）；可写 用户名/名字
    --space NAME        Space 名（默认 multimodal-emotion-demo）
    --private           权重仓库设为私有（这时要在 Space 的 Settings → Secrets 加 HF_TOKEN，否则 Space 下载不了）
    --github URL        GitHub 仓库地址，写进两个 README
    --no-compare        Space 不加载 text-only 对照模型（省约 500 MB 内存）

设计（Day 16 交接里列的问题，各自的处理）：
  - launch 写死 127.0.0.1 → Space 入口是 space\\app.py，自己 launch(0.0.0.0)；app_day16 加了 --host
  - HF_HOME 默认 F:\\hf_cache → space\\app.py 在 import 之前 setdefault 成 ~/.cache
  - 入口要叫 app.py、import 链 8 个脚本 → build 把 src 里 9 个脚本原样复制到 build\\hf_space\\src\\，目录结构和本机一样
  - 权重 1 GB → 单独的 model repo，Space 启动时 hf_hub_download；Space 仓库本身只有几百 KB
  - 不需要 MELD 数据：demo_temperatures.json 带上，T 不用重新拟合；示例只用 space\\examples\\ 里自己录的音频
  - requirements.txt 按本机实际版本生成（和 Day 7 一样用 importlib.metadata），torch 换 CPU 版
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("HF_HOME", r"F:\hf_cache")          # 和其他脚本一样，HF 缓存不占 C 盘
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT_DIR = ROOT / "src", ROOT / "results"
SPACE_SRC = ROOT / "space"                                 # 手写的 Space 文件：app.py + examples\
BUILD = ROOT / "build" / "hf_space"                        # 组装出来、要推送的文件夹（.gitignore 里加 build/）

# app_day16 的 import 链：app_day15 / audio_day8 / fusion_day10 → audio_day9、roberta_day4、text_day5、text_day6、whisper_wer
SCRIPTS = ["app_day16.py", "app_day15.py", "audio_day8.py", "audio_day9.py", "fusion_day10.py",
           "roberta_day4.py", "text_day5.py", "text_day6.py", "whisper_wer.py"]
STEMS = ["fusion_best", "fusion_day13_text", "fusion_day13_audio_e20"]   # 和 app_day16.SPEC 一致
SMALL = ["demo_temperatures.json", "fusion_best_temperature.json", "label_map.json"]
# requirements.txt：import 名 → pip 包名（gradio 不写进去：Space 按 README 的 sdk_version 装）
PKGS = ["transformers", "openai-whisper", "jiwer", "numpy", "pandas", "scikit-learn", "soundfile",
        "matplotlib", "librosa", "huggingface_hub"]


# ================================================================ 工具
def api():
    from huggingface_hub import HfApi
    return HfApi()


def whoami():
    try:
        return api().whoami()["name"]
    except Exception:
        return None


def full_id(name, user):
    return name if "/" in name else f"{user}/{name}"


def version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def need_login():
    user = whoami()
    if not user:
        raise SystemExit("❌ 还没登录 HuggingFace：python -c \"from huggingface_hub import login; login()\"（token 要 Write 权限）")
    return user


def mb(p):
    return p.stat().st_size / 1e6


# ================================================================ check
def cmd_check(a):
    user = whoami()
    print(f"HuggingFace 账号：{user or '❌ 未登录（见脚本开头的第 2 步）'}")
    if user:
        print(f"  权重仓库将是 https://huggingface.co/{full_id(a.model_repo, user)}")
        print(f"  Space 将是     https://huggingface.co/spaces/{full_id(a.space, user)}")
    ok = True
    print("\n要上传 / 复制的文件：")
    for f in [*(f"{s}.pt" for s in STEMS), *(f"{s}.json" for s in STEMS), *SMALL]:
        p = OUT_DIR / f
        need = f != "fusion_best_temperature.json" and not (a.no_compare and f.startswith("fusion_day13_text"))
        print(f"  {'✅' if p.exists() else ('❌' if need else '—')} results\\{f}" + (f"  {mb(p):.1f} MB" if p.exists() else ""))
        ok &= p.exists() or not need
    for f in SCRIPTS:
        p = SRC / f
        print(f"  {'✅' if p.exists() else '❌'} src\\{f}")
        ok &= p.exists()
    p = SPACE_SRC / "app.py"
    print(f"  {'✅' if p.exists() else '❌'} space\\app.py")
    ok &= p.exists()
    ex = [x for x in (SPACE_SRC / "examples").glob("*") if x.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}]
    print(f"  示例音频 space\\examples\\：{len(ex)} 个" + ("（没有也能部署，界面只是没有示例）" if not ex else ""))
    print("\n本机包版本（写进 requirements.txt）：")
    for pkg in ["torch", "gradio", *PKGS]:
        print(f"  {pkg:16s} {version(pkg) or '❌ 没装'}")
    print("\n" + ("检查通过 ✅" if ok and user else "有问题 ⚠️，先处理上面打 ❌ 的项"))


# ================================================================ upload-weights
MODEL_CARD = """---
license: {license}
language: en
tags: [emotion-recognition, speech, multimodal, late-fusion, meld, pytorch]
datasets: [declare-lab/MELD]
base_model: [FacebookAI/roberta-base, facebook/wav2vec2-base]
---

# MELD multimodal emotion recognition: RoBERTa + wav2vec 2.0 late fusion

Weights for the demo Space [{space_id}](https://huggingface.co/spaces/{space_id}).{github}

Seven emotion classes (anger, disgust, fear, joy, neutral, sadness, surprise), trained on MELD (English dialogue from *Friends*).

| file | model | input | dev weighted F1 (seed 42) |
|---|---|---|---|
{rows}

All three share one class (`LateFusionClassifier` in `src/fusion_day10.py`): RoBERTa-base `<s>` vector (fine-tuned, LayerNorm) ‖
learned softmax-weighted sum of the 13 wav2vec2-base hidden states (frozen, mean-pooled over time, LayerNorm)
→ MLP 1536 → 256 → 7. Each `.json` holds the exact `config` and `model_kwargs` used to build the model.

**Test set (MELD test, 2610 utterances, 3 seeds):** text + voice weighted F1 63.9 ± 0.9 vs text-only 62.6 ± 0.7 (same architecture);
paired bootstrap of the 3-seed mean difference +1.3 points, 95% CI [+0.5, +2.0]. Audio-only 45.5 ± 0.3.
Temperature scaling (T fitted on dev) is stored in `demo_temperatures.json`.

**Limitations.** Late fusion is dominated by the text: the same words said in a different tone usually get the same label.
The voice branch only saw *Friends* audio and tends to fall back to *neutral* for new speakers and microphones.
MELD has no sarcasm label. With Whisper-small transcripts instead of gold text, test weighted F1 drops from 63.4 to 52.3 (seed 42).

```python
import json, torch
from fusion_day10 import LateFusionClassifier          # from the GitHub repo
meta = json.load(open("fusion_best.json"))
model = LateFusionClassifier(tuple(meta["config"]["modalities"]), **meta["model_kwargs"])
model.load_state_dict(torch.load("fusion_best.pt", map_location="cpu"))
```
"""


def cmd_upload_weights(a):
    from huggingface_hub import CommitOperationAdd
    user = need_login()
    repo, space_id = full_id(a.model_repo, user), full_id(a.space, user)
    stems = [s for s in STEMS if not (a.no_compare and s == "fusion_day13_text")]
    files = [OUT_DIR / f"{s}{ext}" for s in stems for ext in (".pt", ".json")] + [OUT_DIR / f for f in SMALL]
    files = [p for p in files if p.exists()]
    rows = []
    for s in stems:
        m = json.loads((OUT_DIR / f"{s}.json").read_text(encoding="utf-8"))
        mods = " + ".join(m["config"]["modalities"]).replace("audio", "voice")
        rows.append(f"| `{s}.pt` | epoch {m['best_epoch']} | {mods} | {m['dev_wf1']} |")
    card = MODEL_CARD.format(license=a.license, space_id=space_id, rows="\n".join(rows),
                             github=f" Code: {a.github}" if a.github else "")
    api().create_repo(repo, repo_type="model", private=a.private, exist_ok=True)
    print(f"model repo：https://huggingface.co/{repo}（{'私有' if a.private else '公开'}）")
    for p in files:
        print(f"  {p.name:32s} {mb(p):8.1f} MB")
    print(f"上传 {len(files)} 个文件，共 {sum(mb(p) for p in files):.0f} MB（大文件走 LFS / Xet，断了重跑会跳过已传完的）...")
    t0 = time.time()
    ops = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")))
    api().create_commit(repo, operations=ops, commit_message="Day 17: upload demo weights", repo_type="model")
    print(f"完成，{(time.time() - t0) / 60:.1f} 分钟 ✅  → https://huggingface.co/{repo}/tree/main")


# ================================================================ build
def requirements():
    """本机版本 → requirements.txt。torch 去掉 +cu128 这类后缀，换成 CPU 版（免费 Space 没有 GPU，CPU 版小得多）。"""
    lines = ["# generated by src/deploy_day17.py from the local emotion_ai env (gradio comes from README sdk_version)",
             "--extra-index-url https://download.pytorch.org/whl/cpu"]
    tv = version("torch")
    lines.append(f"torch=={tv.split('+')[0]}+cpu" if tv else "torch")
    for pkg in PKGS:
        v = version(pkg)
        lines.append(f"{pkg}=={v}" if v else pkg)
    return "\n".join(lines) + "\n"


SPACE_README = """---
title: Multimodal Emotion Recognition
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: {gradio}
python_version: "3.11"
app_file: app.py
pinned: false
license: {license}
short_description: Speech emotion recognition with text + voice late fusion
models: [{weights_repo}, FacebookAI/roberta-base, facebook/wav2vec2-base]
datasets: [declare-lab/MELD]
---

# Multimodal emotion recognition (MELD)

Record or upload a short English utterance. Whisper-small transcribes it; a late-fusion model combines
**what was said** (fine-tuned RoBERTa) with **how it was said** (wav2vec 2.0) to predict one of 7 emotions.
The chart compares text + voice, text-only and voice-only models.{github}

- Weights and model card: [{weights_repo}](https://huggingface.co/{weights_repo})
- MELD test weighted F1: 63.9 ± 0.9 (text + voice, 3 seeds) vs 62.6 ± 0.7 (text only)
- Runs on the free CPU tier: the first request after the Space wakes up takes longer; then a few seconds per clip.
- Trained on *Friends* dialogue only. The text usually dominates the fused prediction, so the same words in a
  different tone often keep the same label (the "Tone vs words" note shows when the voice-only model disagrees).
"""


def cmd_build(a):
    user = whoami() or "YOUR_USERNAME"
    repo, space_id = full_id(a.model_repo, user), full_id(a.space, user)
    if not (SPACE_SRC / "app.py").exists():
        raise SystemExit(f"❌ 找不到 {SPACE_SRC / 'app.py'}")
    # 清空重建，但保留 results\*.pt（rehearse 下载过的权重，省得每次重下 1 GB）
    if BUILD.exists():
        for p in BUILD.iterdir():
            if p.name == "results":
                for q in p.iterdir():
                    if q.suffix != ".pt":
                        q.unlink() if q.is_file() else shutil.rmtree(q)
            else:
                p.unlink() if p.is_file() else shutil.rmtree(p)
    (BUILD / "src").mkdir(parents=True, exist_ok=True)
    (BUILD / "results").mkdir(exist_ok=True)

    shutil.copy2(SPACE_SRC / "app.py", BUILD / "app.py")
    for f in SCRIPTS:
        shutil.copy2(SRC / f, BUILD / "src" / f)
    stems = [s for s in STEMS if not (a.no_compare and s == "fusion_day13_text")]
    for f in [f"{s}.json" for s in stems] + SMALL:
        if (OUT_DIR / f).exists():
            shutil.copy2(OUT_DIR / f, BUILD / "results" / f)
    n_ex = 0
    if (SPACE_SRC / "examples").exists():
        shutil.copytree(SPACE_SRC / "examples", BUILD / "examples")
        n_ex = sum(1 for _ in (BUILD / "examples").glob("*") if _.suffix.lower() != ".csv")

    cfg = dict(weights_repo=repo, whisper_model="small", compare=not a.no_compare,
               built_at=time.strftime("%Y-%m-%d %H:%M"), space=space_id)
    (BUILD / "space_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (BUILD / "requirements.txt").write_text(requirements(), encoding="utf-8")
    (BUILD / "packages.txt").write_text("ffmpeg\n", encoding="utf-8")   # apt 包：read_audio 解码 m4a / webm、Gradio 用 ffprobe
    gv = version("gradio")
    if not gv:
        raise SystemExit("❌ 本机没装 gradio？sdk_version 需要它的版本号")
    (BUILD / "README.md").write_text(SPACE_README.format(
        gradio=gv, license=a.license, weights_repo=repo, github=f"\n\nCode: {a.github}" if a.github else ""), encoding="utf-8")

    print(f"已组装 {BUILD}")
    for p in sorted(BUILD.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            tag = "（不上传，Space 启动时从 model repo 下载）" if p.suffix == ".pt" else ""
            print(f"  {str(p.relative_to(BUILD)):45s} {p.stat().st_size / 1e3:9.1f} KB {tag}")
    print(f"\nspace_config.json：weights_repo = {repo}，whisper = small，compare = {not a.no_compare}；示例 {n_ex} 个")
    print("requirements.txt：\n  " + requirements().strip().replace("\n", "\n  "))
    print(f"README sdk_version = gradio {gv}，python 3.11")
    if user == "YOUR_USERNAME":
        print("⚠️ 没登录，仓库名里先写了 YOUR_USERNAME；登录后重新 build")


# ================================================================ rehearse
def cmd_rehearse(a):
    """本机模拟 Space：CUDA_VISIBLE_DEVICES="-1" 强制 CPU；权重从 model repo 下载到 build\\hf_space\\results（真实走一遍下载）。"""
    if not (BUILD / "app.py").exists():
        raise SystemExit("❌ 先运行 build")
    from whisper_wer import DATA, MODEL_DIR
    # Windows 上环境变量不能是空字符串（设成 "" 等于没设），所以用 "-1"：没有编号 -1 的 GPU → torch 看不到 GPU
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", GRADIO_SERVER_NAME="127.0.0.1", GRADIO_SERVER_PORT=str(a.port),
               WHISPER_MODEL_DIR=MODEL_DIR, PYTHONIOENCODING="utf-8")
    if a.launch:
        print(f"演练界面（CPU）：http://127.0.0.1:{a.port}  Ctrl+C 结束")
        cmd = [sys.executable, "app.py"]
    else:
        files = [str(p) for p in a.files] if a.files else []
        if not files:
            ex = sorted(p for p in (BUILD / "examples").glob("*") if p.suffix.lower() != ".csv") if (BUILD / "examples").exists() else []
            meld = [DATA / "audio" / "dev" / f for f in ("dia0_utt0.wav", "dia5_utt0.wav", "dia7_utt1.wav")]
            files = [str(p) for p in ex + [m for m in meld if m.exists()]]   # 示例全跑 + 3 条 MELD dev
        if not files:
            raise SystemExit("❌ 没有可以演练的音频（--files 指定几个 wav）")
        cmd = [sys.executable, "app.py", "--smoke", *files]
    print("运行：" + " ".join(cmd[:3]) + (" ..." if len(cmd) > 3 else "") + f"（在 {BUILD}，CPU only）\n")
    subprocess.run(cmd, cwd=BUILD, env=env, check=False)


# ================================================================ push / status / ping
def cmd_push(a):
    user = need_login()
    space_id = full_id(a.space, user)
    cfg = json.loads((BUILD / "space_config.json").read_text(encoding="utf-8")) if (BUILD / "space_config.json").exists() else None
    if not cfg:
        raise SystemExit("❌ 先运行 build")
    if cfg["weights_repo"].startswith("YOUR_USERNAME"):
        raise SystemExit("❌ build 时没登录，仓库名是 YOUR_USERNAME；登录后重新 build")
    try:
        api().model_info(cfg["weights_repo"])
    except Exception as e:
        raise SystemExit(f"❌ 访问不到权重仓库 {cfg['weights_repo']}（{type(e).__name__}）：先运行 upload-weights")
    api().create_repo(space_id, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    print(f"Space：https://huggingface.co/spaces/{space_id}（公开，默认免费 CPU basic）")
    t0 = time.time()
    api().upload_folder(folder_path=str(BUILD), repo_id=space_id, repo_type="space",
                        ignore_patterns=["*.pt", "**/__pycache__/**", "**/.cache/**"],
                        commit_message=f"Day 17: deploy demo ({time.strftime('%Y-%m-%d %H:%M')})")
    print(f"上传完成（{time.time() - t0:.0f}s）✅。Space 现在开始构建：装依赖 3–8 分钟 + 第一次启动下载模型约 2–3 分钟")
    print(f"  看进度：python src\\deploy_day17.py status   或网页上的 Logs：https://huggingface.co/spaces/{space_id}?logs=container")


def cmd_status(a):
    user = need_login()
    space_id = full_id(a.space, user)
    for i in range(a.wait // 20 + 1):
        rt = api().get_space_runtime(space_id)
        hw = getattr(rt, "hardware", None)
        print(f"  [{time.strftime('%H:%M:%S')}] stage = {rt.stage}  hardware = {hw}")
        if rt.stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "PAUSED", "STOPPED") or not a.wait:
            break
        time.sleep(20)
    sub = space_id.replace("/", "-").replace("_", "-").lower()
    print(f"\n  页面：https://huggingface.co/spaces/{space_id}")
    print(f"  直接打开 app：https://{sub}.hf.space")
    if rt.stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        print(f"  ⚠️ 出错了，看日志：https://huggingface.co/spaces/{space_id}?logs=" + ("build" if rt.stage == "BUILD_ERROR" else "container"))
    elif rt.stage == "RUNNING":
        print("  ✅ 在运行。下一步：python src\\deploy_day17.py ping")


def cmd_ping(a):
    """从外部调用 Space 的 /run 接口 = 别人打开链接、点 Analyse 时走的同一条路（gradio_client 不同版本的 token 参数名不同，这里不传）。"""
    from gradio_client import Client, handle_file
    user = need_login()
    space_id = full_id(a.space, user)
    wav = a.files[0] if a.files else None
    if wav is None:
        ex = sorted(p for p in (BUILD / "examples").glob("*") if p.suffix.lower() != ".csv") if (BUILD / "examples").exists() else []
        from whisper_wer import DATA
        wav = ex[0] if ex else DATA / "audio" / "dev" / "dia0_utt0.wav"
    print(f"连接 {space_id} ...")
    t0 = time.time()
    client = Client(space_id, verbose=False)
    print(f"  连上了（{time.time() - t0:.1f}s；Space 在睡眠的话第一次会等它唤醒）")
    t0 = time.time()
    out = client.predict(handle_file(str(wav)), "Auto-detect", "", "", api_name="/run")
    text, verdict = out[0], out[1]
    print(f"  {Path(wav).name} → {verdict.splitlines()[0].lstrip('# ') if verdict else '(空)'}  转录：{text!r}  （{time.time() - t0:.1f}s）")
    print("✅ Space 能被外部调用。再用手机流量 / 无痕窗口打开一次链接，确认不用登录也能用")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Day 17: HuggingFace Spaces 部署")
    ap.add_argument("cmd", choices=["check", "upload-weights", "build", "rehearse", "push", "status", "ping"])
    ap.add_argument("--model-repo", default="meld-emotion-late-fusion")
    ap.add_argument("--space", default="multimodal-emotion-demo")
    ap.add_argument("--private", action="store_true", help="权重仓库设为私有")
    ap.add_argument("--no-compare", action="store_true", help="不部署 text-only 对照模型")
    ap.add_argument("--license", default="gpl-3.0", help="写进两个 README 的 license（MELD 本身是 GPL-3.0）")
    ap.add_argument("--github", default="", help="GitHub 仓库地址")
    ap.add_argument("--files", nargs="*", type=Path, help="rehearse / ping 用的音频")
    ap.add_argument("--launch", action="store_true", help="rehearse：开界面而不是跑 --smoke")
    ap.add_argument("--port", type=int, default=7861, help="rehearse --launch 的端口（7860 留给 app_day16）")
    ap.add_argument("--wait", type=int, default=0, help="status：最多等几秒直到 RUNNING / 出错（每 20 秒查一次）")
    a = ap.parse_args()
    dict(check=cmd_check, build=cmd_build, rehearse=cmd_rehearse, push=cmd_push, status=cmd_status, ping=cmd_ping,
         **{"upload-weights": cmd_upload_weights})[a.cmd](a)


if __name__ == "__main__":
    main()
