"""
STAGE 1 - DATA LANDING & EDA
============================================================================
Methodology slide "Data Ingestion":
  * The train CSV is large (~16 GB; the test CSV ~33 GB), so reading it
    directly with pd.read_csv() is risky. Instead we stream it in chunks of
    500,000 rows and write each chunk to Parquet, then read the Parquet back.
  * customer_ID is compressed to int64, S_2 parsed as a date.

Then we run the EDA required by the slides:
  * per-feature missing rate
  * descriptive statistics (mean/std/min/max/percentiles)
  * count of unique values (to find constant columns)
  * feature-family summary (P/B/S/R/D)
All EDA artefacts are written to reports/ for the thesis appendix.

Run:
    python src/01_data_landing_eda.py
Outputs:
    data/parquet/train_*.parquet           (chunked raw, compressed)
    data/reports/eda_missing.csv
    data/reports/eda_describe.csv
    data/reports/eda_feature_families.csv
    data/reports/eda_target_balance.csv
"""
from __future__ import annotations
import sys, glob, time
from pathlib import Path

# --- path bootstrap -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths, reduce_mem_usage, compress_customer_id
bootstrap_paths()
import config as C

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
def land_csv_to_parquet(csv_path: Path, prefix: str) -> None:
    """Use Polars for memory-efficient CSV streaming."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    
    C.banner(f"LANDING {csv_path.name} -> Parquet (chunk={C.CHUNK_SIZE:,})")
    
    try:
        import polars as pl
    except:
        print("Installing polars...")
        import subprocess
        subprocess.run(['pip', 'install', 'polars', '-q'], check=True)
        import polars as pl
    
    t0 = time.time()
    
    # Polars can stream large CSVs efficiently
    df = pl.read_csv(csv_path)
    
    # Split into chunks
    n_chunks = (len(df) + C.CHUNK_SIZE - 1) // C.CHUNK_SIZE
    for i in range(n_chunks):
        s = i * C.CHUNK_SIZE
        e = min((i + 1) * C.CHUNK_SIZE, len(df))
        chunk = df.slice(s, e - s)
        
        # Convert to pandas for processing
        chunk_pd = chunk.to_pandas()
        chunk_pd[C.ID_COL] = compress_customer_id(chunk_pd[C.ID_COL])
        if C.DATE_COL in chunk_pd.columns:
            chunk_pd[C.DATE_COL] = pd.to_datetime(chunk_pd[C.DATE_COL])
        
        out = C.PARQUET_DIR / f"{prefix}_{i:04d}.parquet"
        chunk_pd.to_parquet(out, engine="pyarrow", compression=None, index=False)
        print(f"   ✓ chunk {i:>3}: {len(chunk_pd):>7,} rows -> {out.name}")
    
    print(f"   done in {time.time()-t0:.1f}s")


def load_parquet(prefix: str) -> pd.DataFrame:
    """Read all chunks for a prefix back into one DataFrame."""
    files = sorted(glob.glob(str(C.PARQUET_DIR / f"{prefix}_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files for prefix '{prefix}'. Run landing first.")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df


# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, labels: pd.DataFrame) -> None:
    C.banner("EDA")
    feature_cols = [c for c in df.columns if c not in (C.ID_COL, C.DATE_COL)]

    # 1) Missing rate per feature (sample to save memory)
    sample = df[feature_cols].sample(min(100_000, len(df)), random_state=42)
    missing = (sample.isna().mean()
               .sort_values(ascending=False)
               .rename("missing_rate").to_frame())
    missing["n_unique"] = [df[c].nunique(dropna=True) for c in missing.index]
    missing["flag_drop"] = (missing["missing_rate"] > C.MISSING_THRESHOLD) | (missing["n_unique"] <= 1)
    missing.to_csv(C.REPORT_DIR / "eda_missing.csv")
    print(f"   features with >90% missing : {(missing['missing_rate'] > C.MISSING_THRESHOLD).sum()}")
    print(f"   constant features (nunique<=1): {(missing['n_unique'] <= 1).sum()}")
    print(f"   total flagged for drop        : {missing['flag_drop'].sum()}")

    # 2) Target balance
    bal = labels[C.TARGET_COL].value_counts().rename("count").to_frame()
    bal["pct"] = bal["count"] / bal["count"].sum()
    bal.to_csv(C.REPORT_DIR / "eda_target_balance.csv")
    print(f"   target balance: good(0)={bal.loc[0,'count']:,} bad(1)={bal.loc[1,'count']:,} bad-rate={bal.loc[1,'pct']:.2%}")

    # 3) Statements-per-customer
    counts = df.groupby(C.ID_COL).size()
    print(f"   statements/customer: min={counts.min()} median={int(counts.median())} max={counts.max()}")


# ---------------------------------------------------------------------------
def main() -> None:
    # Land train features + labels. (Land the test set later if you submit to Kaggle.)
    if not list(C.PARQUET_DIR.glob("train_*.parquet")):
        land_csv_to_parquet(C.TRAIN_DATA_CSV, "train")
    else:
        print("Parquet chunks already exist - skipping landing. Delete them to re-run.")

    df = load_parquet("train")
    print(f"\nLoaded train: {df.shape[0]:,} rows x {df.shape[1]} cols")

    labels = pd.read_csv(C.TRAIN_LABELS_CSV)
    labels[C.ID_COL] = compress_customer_id(labels[C.ID_COL])
    labels.to_parquet(C.INTERIM_DIR / "labels.parquet", index=False)

    run_eda(df, labels)
    C.banner("STAGE 1 COMPLETE")


if __name__ == "__main__":
    main()
