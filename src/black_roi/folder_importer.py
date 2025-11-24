import os
import io
import imghdr
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

# Allow PIL to load broken/truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_images(input_folder: Path, process_fn: Callable, output_folder_name="output", ocr_results=None) -> list[str]:
    output_folder = input_folder / output_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_path = output_folder / "process_log.txt"
    log_file = open(log_path, "a", encoding="utf-8")

    def log(message: str):
        print(message)
        log_file.write(message + "\n")

    original_stems = []

    # Counters
    processed_count = 0
    skipped_count = 0
    total_images = 0

    for root, dirs, files in os.walk(input_folder):
        root_path = Path(root)

        # Skip output folder
        if output_folder in root_path.parents or root_path == output_folder:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                total_images += 1
                input_path = root_path / file
                relative_path = input_path.relative_to(input_folder)

                stem, suffix = relative_path.stem, relative_path.suffix

                # OCR prefix logic
                ocr_text = ""
                if ocr_results and stem in ocr_results:
                    x_vals = ocr_results[stem].get('X', [])
                    if len(x_vals) >= 2:
                        ocr_text = f"{x_vals[0]}_{x_vals[1]}_"

                processed_name = f"{ocr_text}{stem}{suffix}"
                processed_path = output_folder / relative_path.parent / processed_name
                processed_path.parent.mkdir(parents=True, exist_ok=True)

                # -------- SAFETY CHECK 1: detect if real image ----------
                if imghdr.what(input_path) is None:
                    log(f"⚠️ Not an image despite extension: {input_path}")
                    skipped_count += 1
                    continue

                # -------- SAFETY CHECK 2: attempt opening with recovery ----------
                img = None
                try:
                    with Image.open(input_path) as im:
                        img = im.convert("RGB")  # Fix CMYK, etc.
                except Exception as e:
                    log(f"⚠️ Primary load failed, trying fallback: {input_path} ({e})")

                    # Fallback: read raw bytes
                    try:
                        data = open(input_path, "rb").read()
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        log(f"✔ Recovered using fallback loader: {input_path}")
                    except Exception as e2:
                        log(f"❌ FATAL: Cannot open image even after fallback: {input_path} ({e2})")
                        skipped_count += 1
                        continue

                # -------- PROCESS THE IMAGE ----------
                try:
                    array = np.array(img)
                    processed_array = process_fn(array)
                    Image.fromarray(processed_array).save(processed_path)

                    log(f"🖼️ Saved: {processed_path}")
                    original_stems.append(stem)
                    processed_count += 1

                except Exception as e:
                    log(f"❌ Processing failed for {input_path} ({e})")
                    skipped_count += 1
                    continue

    # -------- SUMMARY ----------
    summary = (
        "\n========== PROCESS SUMMARY ==========\n"
        f"Total images scanned : {total_images}\n"
        f"Successfully processed: {processed_count}\n"
        f"Skipped (errors)     : {skipped_count}\n"
        "=====================================\n"
    )

    print(summary)
    log_file.write(summary)
    log_file.close()

    return original_stems
