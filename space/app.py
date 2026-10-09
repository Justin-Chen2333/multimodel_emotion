"""
Day 17 — HuggingFace Spaces 入口（Space 启动时运行 `python app.py`）

这个文件本身不含模型代码：它只做 Space 特有的几件事，然后调用 Day 16 的 app_day16：
  1. 在 import 任何模型代码之前设好缓存目录（HF_HOME / WHISPER_MODEL_DIR）——src 里的脚本默认 F:\\hf_cache，
     在 Linux 上会变成一个叫 "F:\\hf_cache" 的相对目录，所以这里先 setdefault 成 Linux 的 ~/.cache
  2. 从 HF model repo 下载 3 个 .pt（约 1 GB）到 results\\（.json / demo_temperatures.json 已经在 Space 仓库里）
  3. app_day16.load_all → build_demo（示例用 examples\\ 里自己录的音频，不放 MELD 片段）→ launch(0.0.0.0:7860)

Space 里的目录（由 src\\deploy_day17.py build 生成，和本机项目同一结构，所以 ROOT = parents[1] 那套路径不用改）：
    app.py  space_config.json  requirements.txt  packages.txt  README.md
    src\\     app_day16.py 和它 import 的 8 个脚本（原样复制）
    results\\ *.json（模型配置、温度）+ 启动时下载的 *.pt
    examples\\ 自己录的示例音频 + examples.csv（可选）

本机用法（deploy_day17.py rehearse 会自动这样调用；也可以手动在 build\\hf_space 下运行）：
    python app.py                      # 启动界面（默认 0.0.0.0:7860，和 Space 一样）
    python app.py --smoke a.wav b.wav  # 不开界面：每个文件走一遍界面的 run()，打印结论和用时（CPU 速度演练）
环境变量（都可选；在 Space 的 Settings → Variables 里也能设）：
    WEIGHTS_REPO   权重所在的 model repo（默认读 space_config.json）
    WHISPER_MODEL  默认 small（Day 15 决定：免费 CPU 上 turbo 太慢）
    COMPARE        "0" = 不加载 text-only 对照模型（内存紧张时用）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))                     # 让 `import app_day16` 找得到 src\ 里的脚本
CFG_FILE = HERE / "space_config.json"
CFG = json.loads(CFG_FILE.read_text(encoding="utf-8")) if CFG_FILE.exists() else {}

# ---- 必须在 import app_day16（→ transformers / whisper_wer）之前设置 ----
CACHE = Path.home() / ".cache"
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("WHISPER_MODEL_DIR", str(CACHE / "whisper"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO", CFG.get("weights_repo", ""))
WHISPER = os.environ.get("WHISPER_MODEL", CFG.get("whisper_model", "small"))
COMPARE = os.environ.get("COMPARE", "1" if CFG.get("compare", True) else "0") != "0"
EXAMPLES_DIR = HERE / "examples"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
EXAMPLES_LABEL = "Example clips recorded by the author (try the same words in different tones)"


def fetch_weights(stems):
    """results\\{stem}.pt 不在就从 WEIGHTS_REPO 下载。已经在（本机演练第二次运行 / Space 重启没清盘）就跳过。"""
    from huggingface_hub import hf_hub_download
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    for stem in stems:
        f = f"{stem}.pt"
        if (out / f).exists():
            print(f"  {f} 已在 results\\，跳过下载")
            continue
        if not WEIGHTS_REPO:
            raise SystemExit("❌ 没有 results\\" + f + "，也没设 WEIGHTS_REPO（space_config.json 里的 weights_repo）")
        t0 = time.time()
        print(f"  下载 {WEIGHTS_REPO}/{f} ...", flush=True)
        hf_hub_download(repo_id=WEIGHTS_REPO, filename=f, local_dir=str(out))
        print(f"    完成，{(out / f).stat().st_size / 1e6:.0f} MB，{time.time() - t0:.0f}s", flush=True)


def load_examples():
    """examples\\examples.csv（列：file, transcript）给出顺序和参考文本；没有 csv 就按文件名排序、参考文本留空。
    情绪标签一律留空：界面里的 "gold label" 指 MELD 标注，自己录的句子没有标注。"""
    if not EXAMPLES_DIR.exists():
        return []
    rows = []
    csv = EXAMPLES_DIR / "examples.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv, keep_default_na=False)
        for r in df.itertuples():
            p = EXAMPLES_DIR / str(r.file)
            if p.exists():
                rows.append([str(p), "Auto-detect", str(getattr(r, "transcript", "")), ""])
            else:
                print(f"  ⚠️ examples.csv 里的 {r.file} 不存在，跳过")
    else:
        for p in sorted(EXAMPLES_DIR.iterdir()):
            if p.suffix.lower() in AUDIO_EXT:
                rows.append([str(p), "Auto-detect", "", ""])
    return rows


def smoke(app, files):
    """不开界面，按界面同一条路径（app.run）跑几条音频：看 CPU 上每条要几秒、结论对不对。"""
    secs = []
    for f in files:
        t0 = time.time()
        text, verdict, label, *_ , info = app.run(str(f))
        secs.append(time.time() - t0)
        head = verdict.splitlines()[0].lstrip("# ") if verdict else ""
        print(f"  {Path(f).name:28s} {secs[-1]:5.1f}s  {head:28s} 转录：{text[:60]!r}")
        timing = [l for l in info.split("\n") if l.startswith("**time**")]
        if timing:
            print(f"  {'':28s}        {timing[0].replace('**', '')}")
    if secs:
        print(f"平均每条 {sum(secs) / len(secs):.1f}s（第一条含预热，通常最慢）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", nargs="+", help="不开界面：这些音频各走一遍 run()")
    a = ap.parse_args()

    import torch
    print(f"Python {sys.version.split()[0]} · torch {torch.__version__} · CUDA {torch.cuda.is_available()} · "
          f"CPU 线程 {torch.get_num_threads()} · HF_HOME={os.environ['HF_HOME']}")
    print(f"weights repo = {WEIGHTS_REPO or '（无，只用本地 results）'} · whisper = {WHISPER} · compare = {COMPARE}")
    t0 = time.time()
    fetch_weights(["fusion_best", "fusion_day13_audio_e20"] + (["fusion_day13_text"] if COMPARE else []))

    import app_day16 as app                                 # 这一步才 import transformers / whisper / gradio
    app.load_all(WHISPER, compare=COMPARE)
    print(f"启动总用时 {time.time() - t0:.0f}s")

    if a.smoke:
        smoke(app, a.smoke)
        return

    examples = load_examples()
    demo, _ = app.build_demo(examples=examples, examples_label=EXAMPLES_LABEL)
    print(f"示例音频 {len(examples)} 条")
    # 免费 CPU 只有 2 个核：一次只处理一个请求（default_concurrency_limit=1），最多排队 16 个
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True, allowed_paths=[str(EXAMPLES_DIR)])


if __name__ == "__main__":
    main()
