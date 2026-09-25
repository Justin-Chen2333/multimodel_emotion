"""
把 MELD 的 mp4 批量转成 16kHz 单声道 wav（Wav2Vec 2.0 和 Whisper 都要 16kHz）。

先装 ffmpeg（Whisper 明天也要用）：
    conda install -c conda-forge ffmpeg -y
用法：
    python extract_audio.py            # 全部 split，8 线程
    python extract_audio.py --limit 5  # 每个 split 各转 5 条试试
    python extract_audio.py --split train   # 只转 train
输出：data/MELD/audio/{train,dev,test}/diaX_uttY.wav，已存在的会跳过。
"""
import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "MELD"
OUT = DATA / "audio"


def convert(job):
    src, dst = job
    if dst.exists() and dst.stat().st_size > 0:
        return None
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return None if r.returncode == 0 else (src.name, r.stderr.strip()[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="每个 split 最多转几条")
    ap.add_argument("--split", choices=["train", "dev", "test"], help="只处理这一个 split")
    a = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("没找到 ffmpeg：conda install -c conda-forge ffmpeg -y")

    jobs = []
    for mp4 in DATA.rglob("*.mp4"):
        if mp4.name.startswith("._"):
            continue
        folder = mp4.parent.name.lower()
        split = "train" if "train" in folder else "dev" if "dev" in folder else "test" if "test" in folder else None
        if split is None or (a.split and split != a.split):
            continue
        (OUT / split).mkdir(parents=True, exist_ok=True)
        jobs.append((mp4, OUT / split / mp4.with_suffix(".wav").name))
    jobs.sort()
    if a.limit:  # 每个 split 各取前 N 条，而不是总共前 N 条
        kept, seen = [], {}
        for src, dst in jobs:
            sp = dst.parent.name
            if seen.get(sp, 0) < a.limit:
                kept.append((src, dst))
                seen[sp] = seen.get(sp, 0) + 1
        jobs = kept
    from collections import Counter
    print(f"共 {len(jobs)} 个视频待处理：", dict(Counter(d.parent.name for _, d in jobs)))

    fails = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(convert, jobs), 1):
            if res:
                fails.append(res)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"完成，失败 {len(fails)} 个")
    for name, err in fails[:10]:
        print("  ", name, err)


if __name__ == "__main__":
    main()
