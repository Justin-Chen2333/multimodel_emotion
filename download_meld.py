"""
Day 2 — 下载并校验 MELD（Windows / Anaconda Prompt 下运行）

用法（在 F:\\condaEnv\\multimodel-emotion 下，conda activate emotion_ai 之后）：
    python download_meld.py --csv-only        # 1 分钟内拿到文本+标签，完成今日最低目标
    python download_meld.py                   # 再下载 10.9GB 原始视频包并解压（可断点续传）
    python download_meld.py --mirror          # 在国内网络：走 hf-mirror.com
    python download_meld.py --skip-download   # 已经手动下载好 tar.gz，只解压+校验
    python download_meld.py --tar "路径\\MELD.Raw.tar.gz"   # 指定已有的压缩包，不再复制一份

注意：load_dataset("declare-lab/MELD") 不能用——那个 HF 仓库里只有两个 tar.gz，
没有 parquet/加载脚本，datasets 无法解析。所以这里直接下载文件。
"""
import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "MELD"
CSV_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{split}_sent_emo.csv"
SPLITS = ["train", "dev", "test"]
EXPECTED_ROWS = {"train": 9989, "dev": 1109, "test": 2610}


def get_csvs():
    csv_dir = DATA / "annotations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        dst = csv_dir / f"{s}_sent_emo.csv"
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[csv] {dst.name} 已存在，跳过")
            continue
        print(f"[csv] 下载 {dst.name} ...")
        urllib.request.urlretrieve(CSV_URL.format(split=s), dst)
    return csv_dir


def check_csvs(csv_dir):
    import pandas as pd
    for s in SPLITS:
        df = pd.read_csv(csv_dir / f"{s}_sent_emo.csv", encoding="utf-8")
        flag = "OK" if len(df) == EXPECTED_ROWS[s] else f"!! 期望 {EXPECTED_ROWS[s]}"
        print(f"\n[{s}] {len(df)} 条  {flag}")
        print(df["Emotion"].value_counts(normalize=True).round(3).to_string())
    df = pd.read_csv(csv_dir / "train_sent_emo.csv")
    r = df.iloc[0]
    print("\n样本示例：", {k: r[k] for k in ["Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion", "Sentiment"]})
    print(f"对应视频文件名：dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4")


def download_raw(mirror):
    if mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import hf_hub_download  # 必须在设置 HF_ENDPOINT 之后 import
    kw = dict(repo_id="declare-lab/MELD", repo_type="dataset", filename="MELD.Raw.tar.gz")
    try:
        # 如果 load_dataset 之前已经下过，直接用 HF 缓存里的那份，不重复下载、不复制
        path = hf_hub_download(**kw, local_files_only=True)
        print(f"[raw] 在 HF 缓存里找到了：{path}")
    except Exception:
        print(f"[raw] 从 {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')} 下载 MELD.Raw.tar.gz (10.9GB)，中断后重跑即可续传")
        path = hf_hub_download(**kw)  # 存到 HF_HOME 指向的缓存目录
    return Path(path)


def safe_extract(tar_path, dest):
    print(f"[extract] {tar_path.name} -> {dest}")
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            tf.extractall(dest, filter="data")  # Python 3.11.4+
        except TypeError:
            tf.extractall(dest)


def extract_all(outer=None):
    outer = Path(outer) if outer else DATA / "MELD.Raw.tar.gz"
    if not outer.exists():
        sys.exit(f"找不到 {outer}，先下载或手动放到这个位置")
    marker = DATA / ".outer_extracted"
    if not marker.exists():
        safe_extract(outer, DATA)
        marker.touch()
    # 外层包里还套着 train/dev/test 三个 tar.gz
    for inner in sorted(DATA.rglob("*.tar.gz")):
        if inner.resolve() == outer.resolve():
            continue
        m = inner.with_suffix(".extracted")
        if m.exists():
            continue
        safe_extract(inner, inner.parent)
        m.touch()


def check_videos(csv_dir):
    import pandas as pd
    all_mp4 = {}
    for p in DATA.rglob("*.mp4"):
        if p.name.startswith("._"):  # macOS 打包留下的垃圾文件
            continue
        all_mp4.setdefault(p.parent.name, []).append(p)
    print("\n[video] 各文件夹 mp4 数量：")
    for folder, files in sorted(all_mp4.items()):
        print(f"  {folder}: {len(files)}")
    # 按 split 对齐 CSV 与视频
    hint = {"train": "train", "dev": "dev", "test": "test"}
    for s in SPLITS:
        folders = [f for f in all_mp4 if hint[s] in f.lower()]
        names = {p.name for f in folders for p in all_mp4[f]}
        df = pd.read_csv(csv_dir / f"{s}_sent_emo.csv")
        want = {f"dia{d}_utt{u}.mp4" for d, u in zip(df.Dialogue_ID, df.Utterance_ID)}
        missing = sorted(want - names)
        print(f"[{s}] CSV {len(want)} 条，找到视频 {len(want) - len(missing)}，缺失 {len(missing)}"
              + (f"，例如 {missing[:3]}" if missing else ""))
    print("\n少量缺失（个位数）是 MELD 已知情况，训练时跳过即可。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="使用 hf-mirror.com（国内网络）")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--tar", help="已有的 MELD.Raw.tar.gz 路径")
    a = ap.parse_args()

    csv_dir = get_csvs()
    check_csvs(csv_dir)
    if a.csv_only:
        sys.exit(0)
    tar = a.tar
    if tar is None and not a.skip_download:
        tar = download_raw(a.mirror)
    extract_all(tar)
    check_videos(csv_dir)
