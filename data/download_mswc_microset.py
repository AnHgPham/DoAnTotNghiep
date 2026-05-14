"""Download and prepare the official MLCommons MSWC Microset.

The Microset is a small official subset of MSWC (English + Spanish) intended
for prototyping. It is useful when Colab/local disk is not enough for MSWC
English Top500/full extraction.

Usage:
    python data/download_mswc_microset.py --language en
    python data/download_mswc_microset.py --language es
    python data/download_mswc_microset.py --language en --skip-convert
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

MICROSET_URLS = {
    "cloudflare": "https://mswc.mlcommons-storage.org/mswc_microset.tar.gz",
    "alibaba": "https://mlc-datasets.oss-cn-guangzhou.aliyuncs.com/mswc_microset.tar.gz",
    "google": "https://storage.googleapis.com/public-datasets-mswc/mswc_microset.tar.gz",
}


def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(1, retries + 1):
        resume_pos = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        try:
            with requests.get(url, stream=True, timeout=(20, 180), headers=headers) as resp:
                if resp.status_code == 416:
                    tmp.replace(dest)
                    return
                resp.raise_for_status()
                if resume_pos and resp.status_code != 206:
                    logger.warning("Server ignored Range header; restarting download for %s", dest)
                    resume_pos = 0
                total = int(resp.headers.get("content-length", 0)) + resume_pos
                mode = "ab" if resume_pos else "wb"
                with open(tmp, mode) as f, tqdm(
                    total=total,
                    initial=resume_pos,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as pbar:
                    for chunk in resp.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 5 * attempt
            logger.warning("Download failed (%s), retrying in %ss", exc, wait)
            time.sleep(wait)


def download_microset(archive: Path, mirror: str = "cloudflare") -> Path:
    if archive.exists():
        logger.info("Archive already exists: %s (%.1f MB)", archive, archive.stat().st_size / 1024**2)
        return archive

    mirrors = [mirror] + [m for m in MICROSET_URLS if m != mirror]
    last_error: Exception | None = None
    for name in mirrors:
        try:
            logger.info("Downloading MSWC Microset from %s mirror", name)
            _download_file(MICROSET_URLS[name], archive)
            return archive
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Mirror %s failed: %s", name, exc)
            archive.with_suffix(archive.suffix + ".partial").unlink(missing_ok=True)

    raise RuntimeError(f"All Microset mirrors failed: {last_error}")


def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> bool:
    target_resolved = target.resolve()
    out_path = (target / member.name).resolve()
    if not str(out_path).startswith(str(target_resolved)):
        logger.warning("Skipping unsafe tar member: %s", member.name)
        return False
    if member.isdir():
        out_path.mkdir(parents=True, exist_ok=True)
        return True
    if not member.isfile():
        logger.debug("Skipping non-file tar member: %s", member.name)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    if src is None:
        return False
    with src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return True


def extract_language(archive: Path, language: str, output_dir: Path, force: bool = False) -> None:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"mswc_microset/{language}/"
    extracted = 0
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
        if not members:
            raise RuntimeError(f"No members found for language={language!r} in {archive}")
        for member in tqdm(members, desc=f"Extracting Microset {language}"):
            rel = member.name[len(prefix):]
            if not rel:
                continue
            member.name = rel
            if _safe_extract_member(tar, member, output_dir):
                extracted += 1
    logger.info("Extracted %d members to %s", extracted, output_dir)


def write_word_splits(output_dir: Path, val_fraction: float, seed: int) -> None:
    clips = output_dir / "clips"
    words = sorted(
        d.name
        for d in clips.iterdir()
        if d.is_dir() and (any(d.glob("*.wav")) or any(d.glob("*.opus")))
    )
    if len(words) < 2:
        raise RuntimeError(f"Need at least 2 words, found {len(words)} in {clips}")

    rng = random.Random(seed)
    shuffled = words.copy()
    rng.shuffle(shuffled)
    n_val = max(1, min(round(len(words) * val_fraction), len(words) - 1))
    val_words = sorted(shuffled[:n_val])
    train_words = sorted(shuffled[n_val:])

    splits = output_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "train_words.json").write_text(json.dumps(train_words, indent=2), encoding="utf-8")
    (splits / "val_words.json").write_text(json.dumps(val_words, indent=2), encoding="utf-8")
    (splits / "eval_words.json").write_text("[]", encoding="utf-8")
    logger.info("Splits: %d train, %d val -> %s", len(train_words), len(val_words), splits)


def convert_opus(output_dir: Path, workers: int, delete_opus: bool) -> None:
    clips = output_dir / "clips"
    opus_n = sum(1 for _ in clips.rglob("*.opus"))
    if opus_n == 0:
        logger.info("No OPUS files found; conversion skipped")
        return

    try:
        subprocess.run(
            ["apt-get", "install", "-qq", "ffmpeg"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass

    cmd = [
        sys.executable,
        "data/convert_opus.py",
        "--clips-dir",
        str(clips),
        "--workers",
        str(workers),
    ]
    if delete_opus:
        cmd.append("--delete-opus")
    subprocess.run(cmd, check=True)


def summarize(output_dir: Path) -> None:
    clips = output_dir / "clips"
    wav_n = sum(1 for _ in clips.rglob("*.wav"))
    opus_n = sum(1 for _ in clips.rglob("*.opus"))
    words = sorted(d for d in clips.iterdir() if d.is_dir())
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1024**2
    logger.info("Microset ready: %d words, %d WAV, %d OPUS, %.1f MB at %s",
                len(words), wav_n, opus_n, total_size, output_dir)


def _has_existing_audio(output_dir: Path) -> bool:
    clips = output_dir / "clips"
    return clips.exists() and (
        any(clips.rglob("*.wav")) or any(clips.rglob("*.opus"))
    )


def _has_splits(output_dir: Path) -> bool:
    splits = output_dir / "splits"
    return (
        (splits / "train_words.json").exists()
        and (splits / "val_words.json").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MLCommons MSWC Microset")
    parser.add_argument("--language", choices=["en", "es"], default="en")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=Path("data/mswc_microset.tar.gz"))
    parser.add_argument("--mirror", choices=list(MICROSET_URLS), default="cloudflare")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--keep-opus", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = args.output_dir or Path(f"data/mswc_microset_{args.language}")
    if _has_existing_audio(output_dir) and not args.force:
        logger.info("Existing Microset audio found at %s; skipping archive download/extract", output_dir)
    else:
        archive = download_microset(args.archive, mirror=args.mirror)
        extract_language(archive, args.language, output_dir, force=args.force)
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
            logger.info("Deleted archive: %s", archive)

    if not args.skip_convert:
        convert_opus(output_dir, workers=args.workers, delete_opus=not args.keep_opus)
    if not _has_splits(output_dir) or args.force:
        write_word_splits(output_dir, val_fraction=args.val_fraction, seed=args.seed)
    else:
        logger.info("Existing Microset splits found at %s", output_dir / "splits")
    summarize(output_dir)


if __name__ == "__main__":
    main()
