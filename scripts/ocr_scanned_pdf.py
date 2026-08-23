#!/usr/bin/env python3
"""Resume-capable parallel OCR for scanned Chinese/English PDFs."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import tempfile
from pathlib import Path


def ocr_page(job: tuple[str, str, int, int, str, str]) -> tuple[int, int]:
    pdf, output_dir, page, dpi, tesseract, languages = job
    target = Path(output_dir) / f"page_{page:04d}.txt"
    if target.is_file() and target.stat().st_size > 20:
        return page, target.stat().st_size
    with tempfile.TemporaryDirectory(prefix=f"ocr_p{page:04d}_") as temp_dir:
        image_base = Path(temp_dir) / "page"
        subprocess.run(
            [
                "pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi),
                "-gray", "-singlefile", "-png", pdf, str(image_base),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        text_base = Path(temp_dir) / "text"
        subprocess.run(
            [
                tesseract, str(image_base.with_suffix(".png")), str(text_base),
                "-l", languages, "--psm", "6",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        text = text_base.with_suffix(".txt").read_text("utf-8", errors="strict")
        target.write_text(text, encoding="utf-8")
    return page, target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pages", type=int, required=True)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--languages", default="chi_sim+eng")
    parser.add_argument(
        "--tesseract",
        default="tesseract",
        help="Tesseract executable name or path; defaults to PATH lookup.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(args.pdf), str(args.output_dir), page, args.dpi,
            args.tesseract, args.languages,
        )
        for page in range(1, args.pages + 1)
    ]
    completed = total_bytes = 0
    with concurrent.futures.ProcessPoolExecutor(args.workers) as executor:
        futures = [executor.submit(ocr_page, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            page, byte_count = future.result()
            completed += 1
            total_bytes += byte_count
            if completed % 25 == 0 or completed == len(jobs):
                print(
                    f"completed={completed}/{len(jobs)} "
                    f"last_page={page} bytes={total_bytes}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
