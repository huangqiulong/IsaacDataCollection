#!/usr/bin/env python3
"""RLDS / Open X-Embodiment 数据集下载助手（纯标准库，无需 gsutil）。

从 Google Cloud 公共桶 gresearch/robotics 直接并行拉取某个数据集的全部分片，
落盘到 TFDS 目录布局：<dest>/<dataset_name>/<version>/...

用法
----
    python3 rlds_download.py <dataset_name> [--version 0.1.0] \
        [--dest ~/tensorflow_datasets] [--workers 16]

示例
----
    # 86MB 的小数据集（本文演示用）
    python3 rlds_download.py imperial_wrist_dataset --version 1.0.0 \
        --dest /media/你的盘/datasets/rlds_tfds

特性：断点续传（按远端 size 校验跳过已完成分片）、逐文件重试、多线程并行。
注意：GCS 直连速度一般（单连接 ~150KB/s，16 路并行约 0.5-1MB/s）；
      若需要走代理，脚本会遵循 https_proxy / ALL_PROXY 环境变量
      （注意：需要 socks 支持时先安装 pysocks，或改用 http 代理）。
"""
from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "gresearch"
API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
FILE_API = "https://storage.googleapis.com/{bucket}/{name}"
CHUNK_BYTES = 8 << 20        # 大文件分片大小（8MB）
CHUNK_THRESHOLD = 32 << 20   # 超过该大小的文件采用分片并行


def _get_json(url: str, tries: int = 5) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rlds-download/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                import json
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return {}


def list_objects(prefix: str) -> list[dict]:
    """列出桶内 prefix 下所有对象 [{name, size}]。"""
    items, token = [], None
    while True:
        url = (f"{API.format(bucket=BUCKET)}?prefix={urllib.parse.quote(prefix)}"
               f"&maxResults=1000&fields=items(name,size),nextPageToken")
        if token:
            url += f"&pageToken={token}"
        d = _get_json(url)
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            break
    return [x for x in items if not x["name"].endswith("_$folder$")]


def fetch_range(remote: str, start: int, end: int, dest: Path, tries: int = 4) -> str:
    """下载 [start, end] 字节区间到 dest（含断点续传与重试）。"""
    want = end - start + 1
    if dest.exists() and dest.stat().st_size == want:
        return "SKIP"
    url = FILE_API.format(bucket=BUCKET, name=urllib.parse.quote(remote))
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "rlds-download/1.0",
                "Range": f"bytes={start}-{end}",
            })
            with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if dest.stat().st_size != want:
                raise IOError(f"size mismatch {dest.stat().st_size} != {want}")
            return "OK"
        except Exception as e:
            if i == tries - 1:
                return f"FAIL {dest.name}: {e}"
            time.sleep(2 * (i + 1))
    return f"FAIL {dest.name}"


def assemble(local: Path, size: int) -> bool:
    """把所有分片按序拼回 local，并删除分片。"""
    parts = sorted(local.parent.glob(local.name + ".part[0-9][0-9][0-9]"))
    if not parts or sum(p.stat().st_size for p in parts) != size:
        return False
    tmp = local.with_suffix(local.suffix + ".tmp")
    with tmp.open("wb") as out:
        for p in parts:
            out.write(p.read_bytes())
            p.unlink()
    if tmp.stat().st_size != size:
        tmp.unlink()
        return False
    tmp.rename(local)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", help="数据集名，如 imperial_wrist_dataset")
    ap.add_argument("--version", default="0.1.0", help="版本目录（默认 0.1.0）")
    ap.add_argument("--dest", default=str(Path.home() / "tensorflow_datasets"),
                    help="TFDS 数据根目录（默认 ~/tensorflow_datasets）")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prefix-root", default="robotics",
                    help="桶内根前缀（默认 robotics）")
    args = ap.parse_args()

    prefix = f"{args.prefix_root}/{args.dataset}/{args.version}/"
    out = Path(args.dest) / args.dataset / args.version
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] 列出对象: gs://{BUCKET}/{prefix}")
    objs = list_objects(prefix)
    total = sum(int(x["size"]) for x in objs)
    print(f"      {len(objs)} 个文件，共 {total/1e6:.1f} MB -> {out}")

    print(f"[2/2] 并行下载（{args.workers} 路；>32MB 自动分片）...")
    t0 = time.time()

    # 展开任务：每个文件 1~N 个分片（小文件=单分片）
    tasks = []
    for x in objs:
        remote, size = x["name"], int(x["size"])
        local = out / remote[len(prefix):]
        local.parent.mkdir(parents=True, exist_ok=True)
        if size <= CHUNK_THRESHOLD:
            tasks.append((remote, 0, size - 1, local))
        else:
            for i, start in enumerate(range(0, size, CHUNK_BYTES)):
                end = min(size, start + CHUNK_BYTES) - 1
                tasks.append((remote, start, end, local.with_name(local.name + f".part{i:03d}")))

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_range, r, s, e, d): d for r, s, e, d in tasks}
        for fut in as_completed(futs):
            msg = fut.result()
            if msg.startswith("FAIL"):
                print("   ", msg, flush=True)
            else:
                n_done += 1

    # 拼装分片 & 校验
    final_bad = []
    for x in objs:
        remote, size = x["name"], int(x["size"])
        local = out / remote[len(prefix):]
        if local.exists() and local.stat().st_size == size:
            print(f"   OK   {local.name} ({size/1e6:.1f} MB)")
        elif assemble(local, size):
            print(f"   OK   {local.name} ({size/1e6:.1f} MB, 分片拼装)")
        else:
            final_bad.append(local.name)

    ok = len(objs) - len(final_bad)
    print(f"完成：{ok}/{len(objs)} 个文件（{n_done}/{len(tasks)} 个分片任务），用时 {time.time()-t0:.0f}s")
    if final_bad:
        print("失败文件（可重跑本脚本续传）：", final_bad)
    print(f"提示：读取时设置 TFDS_DATA_DIR={args.dest}（或 tfds.load(..., data_dir=...)）")


if __name__ == "__main__":
    main()
