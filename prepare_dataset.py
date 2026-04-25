"""
prepare_dataset.py
==================
Run this ONCE before your pipeline to turn all your Merged CSV files
into one clean, balanced dataset ready for the project.

Usage:
    python prepare_dataset.py --input_dir "path/to/your/merged/csvs" --output "data/raw/ciciot2023_sample.csv"

Example:
    python prepare_dataset.py --input_dir "C:/Users/you/Downloads/CICIoT" --output "data/raw/ciciot2023_sample.csv"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_dataset")

# ── Configuration ──────────────────────────────────────────────────────────────

# How many rows to sample PER FILE (tune this based on your RAM)
# At ~130MB per file and 60 files: 3000 rows/file → ~180,000 total rows (good)
# Increase to 5000 if you have 16GB+ RAM, decrease to 1500 if low on memory
ROWS_PER_FILE = 3000

# The column that holds the attack label — CICIoT2023 common names:
CANDIDATE_TARGET_COLS = ["label", "Label", "attack_type", "Attack_type", "Attack", "Class"]

# Columns that are known ID/metadata leaks in CICIoT2023 — will be dropped
KNOWN_LEAKY_COLS = [
    "flow_id", "Flow ID", "Timestamp", "timestamp",
    "src_ip", "dst_ip", "Src IP", "Dst IP",
    "src_port", "dst_port", "Src Port", "Dst Port",
    "Protocol",
]

# ── Core functions ─────────────────────────────────────────────────────────────

def detect_target_column(df: pd.DataFrame) -> str | None:
    for col in CANDIDATE_TARGET_COLS:
        if col in df.columns:
            return col
    return None


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    original_len = len(df)

    # 1. Drop fully empty rows
    df = df.dropna(how="all")

    # 2. Replace infinities with NaN so imputer can handle them
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 3. Drop duplicate rows (CICIoT2023 has many)
    df = df.drop_duplicates()

    # 4. Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # 5. Strip whitespace from target labels (common in CICIoT exports)
    df[target_col] = df[target_col].astype(str).str.strip()

    # 6. Drop known leaky/metadata columns that are present
    cols_to_drop = [c for c in KNOWN_LEAKY_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 7. Drop near-constant columns (unique values <= 1 across the chunk)
    near_const = [c for c in df.columns if c != target_col and df[c].nunique(dropna=False) <= 1]
    if near_const:
        df = df.drop(columns=near_const)

    cleaned_len = len(df)
    removed = original_len - cleaned_len
    if removed > 0:
        log.debug(f"  Cleaned {removed} rows ({removed/original_len:.1%} of chunk)")

    return df


def stratified_sample(df: pd.DataFrame, target_col: str, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows with proportional class representation."""
    class_counts = df[target_col].value_counts()
    n = min(n, len(df))

    samples = []
    for cls, count in class_counts.items():
        frac = count / len(df)
        cls_n = max(1, int(n * frac))
        cls_df = df[df[target_col] == cls]
        cls_sample = cls_df.sample(min(cls_n, len(cls_df)), random_state=seed)
        samples.append(cls_sample)

    result = pd.concat(samples, ignore_index=True)

    # Trim to exactly n if we oversampled slightly
    if len(result) > n:
        result = result.sample(n, random_state=seed).reset_index(drop=True)

    return result


def process_all_files(
    input_dir: Path,
    output_path: Path,
    rows_per_file: int = ROWS_PER_FILE,
    seed: int = 42,
) -> None:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        log.error(f"No CSV files found in {input_dir}")
        sys.exit(1)

    log.info(f"Found {len(csv_files)} CSV files in {input_dir}")
    log.info(f"Sampling {rows_per_file} rows per file → target ~{len(csv_files) * rows_per_file:,} total rows")

    all_chunks: list[pd.DataFrame] = []
    target_col: str | None = None
    failed_files: list[str] = []

    for i, fpath in enumerate(csv_files, 1):
        log.info(f"[{i}/{len(csv_files)}] Processing {fpath.name} ...")

        try:
            # Read only a chunk to save memory (3x rows_per_file gives us room to clean + sample)
            df = pd.read_csv(fpath, nrows=rows_per_file * 3, low_memory=False)
        except Exception as e:
            log.warning(f"  Could not read {fpath.name}: {e} — skipping")
            failed_files.append(fpath.name)
            continue

        # Auto-detect target column from first file
        if target_col is None:
            target_col = detect_target_column(df)
            if target_col is None:
                log.error(
                    f"Could not find a label column in {fpath.name}.\n"
                    f"Columns found: {list(df.columns)}\n"
                    f"Edit CANDIDATE_TARGET_COLS in this script to match your column name."
                )
                sys.exit(1)
            log.info(f"  Detected target column: '{target_col}'")

        df = clean_dataframe(df, target_col)

        if len(df) == 0:
            log.warning(f"  {fpath.name} is empty after cleaning — skipping")
            failed_files.append(fpath.name)
            continue

        sample = stratified_sample(df, target_col, rows_per_file, seed)
        all_chunks.append(sample)

        classes = sample[target_col].value_counts().to_dict()
        log.info(f"  Sampled {len(sample)} rows | Classes: {classes}")

    if not all_chunks:
        log.error("No data collected. Check your input directory and CSV format.")
        sys.exit(1)

    log.info("Merging all chunks ...")
    merged = pd.concat(all_chunks, ignore_index=True)

    # Final shuffle so files aren't in order
    merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Final dedup across the merged set
    before = len(merged)
    merged = merged.drop_duplicates()
    after = len(merged)
    if before != after:
        log.info(f"Removed {before - after} cross-file duplicates")

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Final dataset: {len(merged):,} rows × {merged.shape[1]} columns")
    log.info(f"Target column: '{target_col}'")
    log.info("Class distribution:")
    for cls, cnt in merged[target_col].value_counts().items():
        pct = cnt / len(merged) * 100
        log.info(f"  {cls:<35} {cnt:>7,} rows ({pct:.1f}%)")

    if failed_files:
        log.warning(f"Skipped {len(failed_files)} files: {failed_files}")

    # ── Save ───────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_048_576
    log.info(f"Saved to: {output_path}  ({size_mb:.1f} MB)")
    log.info("=" * 60)
    log.info("Next step: run  python scripts/run_all.py")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CICIoT2023 dataset for the IDS pipeline.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing your Merged*.csv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/ciciot2023_sample.csv",
        help="Where to save the cleaned merged CSV (default: data/raw/ciciot2023_sample.csv)",
    )
    parser.add_argument(
        "--rows_per_file",
        type=int,
        default=ROWS_PER_FILE,
        help=f"Rows to sample per CSV file (default: {ROWS_PER_FILE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_files(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        rows_per_file=args.rows_per_file,
        seed=args.seed,
    )
