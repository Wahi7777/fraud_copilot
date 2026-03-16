from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
SAMPLE_DIR = DATA_DIR / "vercel_sample"
PAYSIM_FILE = "PS_20174392719_1491204439457_log.csv"
IEEE_TX_FILE = "train_transaction.csv"
IEEE_ID_FILE = "train_identity.csv"
SAMPLE_ROWS = 10_000


def _read_head(path: Path, n: int) -> pd.DataFrame:
    if not path.exists():
        print(f"[create_vercel_sample] WARNING: source file not found: {path}")
        return pd.DataFrame()
    print(f"[create_vercel_sample] Reading head({n}) from {path}")
    return pd.read_csv(path, nrows=n)


def create_paysim_sample() -> None:
    src = DATA_DIR / PAYSIM_FILE
    dst = SAMPLE_DIR / PAYSIM_FILE
    df = _read_head(src, SAMPLE_ROWS)
    if df.empty:
        print("[create_vercel_sample] Skipping PaySim sample – source is empty or missing.")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"[create_vercel_sample] Wrote PaySim sample to {dst} with {len(df)} rows")


def create_ieee_sample() -> None:
    tx_src = DATA_DIR / IEEE_TX_FILE
    id_src = DATA_DIR / IEEE_ID_FILE
    tx_df = _read_head(tx_src, SAMPLE_ROWS)
    if tx_df.empty:
        print("[create_vercel_sample] Skipping IEEE samples – transaction source is empty or missing.")
        return

    # Filter identity rows to the sampled TransactionID set when possible.
    if id_src.exists():
        print("[create_vercel_sample] Filtering identity rows for sampled TransactionID set.")
        id_full = pd.read_csv(id_src)
        tx_ids = set(tx_df["TransactionID"].tolist())
        id_df = id_full[id_full["TransactionID"].isin(tx_ids)]
        if id_df.empty:
            id_df = id_full.head(SAMPLE_ROWS)
    else:
        print("[create_vercel_sample] Identity source missing; using empty identity sample.")
        id_df = pd.DataFrame()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    tx_dst = SAMPLE_DIR / IEEE_TX_FILE
    id_dst = SAMPLE_DIR / IEEE_ID_FILE
    tx_df.to_csv(tx_dst, index=False)
    id_df.to_csv(id_dst, index=False)
    print(f"[create_vercel_sample] Wrote IEEE transaction sample to {tx_dst} with {len(tx_df)} rows")
    print(f"[create_vercel_sample] Wrote IEEE identity sample to {id_dst} with {len(id_df)} rows")


def main() -> None:
    print("[create_vercel_sample] Using DATA_DIR:", DATA_DIR.resolve())
    mode = os.getenv("DATA_MODE", "sample")
    print(f"[create_vercel_sample] DATA_MODE={mode}")
    create_paysim_sample()
    create_ieee_sample()
    print("[create_vercel_sample] Done.")


if __name__ == "__main__":
    main()

