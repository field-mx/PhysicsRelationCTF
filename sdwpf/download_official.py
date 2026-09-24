"""Parallel ranged downloader for the immutable official Figshare file."""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


FILE_ID = 46005810
SIZE = 291_653_253
MD5 = "16e40b93890d43a18946bafbd6aa3577"
CHUNKS = 8
TARGET = Path(__file__).resolve().parents[1] / "data/sdwpf/sdwpf_2001_2112_full.parquet"


def signed_url() -> str:
    response = requests.get(
        f"https://ndownloader.figshare.com/files/{FILE_ID}",
        allow_redirects=False,
        timeout=30,
    )
    response.raise_for_status()
    return response.headers["location"]


def download_one(args: tuple[int, int, int, str]) -> Path:
    index, start, end, url = args
    part = TARGET.with_suffix(f".part{index:02d}")
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return part
    with requests.get(
        url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 300)
    ) as response:
        if response.status_code != 206:
            raise RuntimeError(f"range {index}: HTTP {response.status_code}")
        with part.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if part.stat().st_size != expected:
        raise RuntimeError(f"range {index}: short download")
    return part


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    url = signed_url()
    ranges = []
    for index in range(CHUNKS):
        start = index * SIZE // CHUNKS
        end = (index + 1) * SIZE // CHUNKS - 1
        ranges.append((index, start, end, url))
    with ThreadPoolExecutor(max_workers=CHUNKS) as pool:
        parts = list(pool.map(download_one, ranges))
    assembled = TARGET.with_suffix(".assembling")
    with assembled.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
    digest = hashlib.md5(assembled.read_bytes()).hexdigest()
    if digest != MD5:
        raise RuntimeError(f"MD5 mismatch: {digest}")
    os.replace(assembled, TARGET)
    for part in parts:
        part.unlink()
    print(f"verified {TARGET} md5={digest}")


if __name__ == "__main__":
    main()
