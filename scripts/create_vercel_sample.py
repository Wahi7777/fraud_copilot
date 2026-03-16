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
    if not src.exists():
        print(f"[create_vercel_sample] Skipping PaySim sample – source missing at {src}")
        return
    print(f"[create_vercel_sample] Reading full PaySim from {src} (this may take a moment)...")
    df = pd.read_csv(src)
    if df.empty or "isFraud" not in df.columns:
        print("[create_vercel_sample] PaySim source has no rows or no isFraud column; using head-only sample.")
        df_sample = df.head(SAMPLE_ROWS)
    else:
        fraud_df = df[df["isFraud"] == 1]
        non_fraud_df = df[df["isFraud"] == 0]
        n_fraud = min(len(fraud_df), SAMPLE_ROWS // 4) or min(len(fraud_df), 1000)
        n_non_fraud = min(len(non_fraud_df), SAMPLE_ROWS - n_fraud)
        fraud_part = fraud_df.head(n_fraud)
        non_fraud_part = non_fraud_df.head(n_non_fraud)
        df_sample = pd.concat([fraud_part, non_fraud_part], ignore_index=True)
        print(
            f"[create_vercel_sample] PaySim source rows={len(df)} fraud_total={len(fraud_df)} "
            f"sample_rows={len(df_sample)} sample_fraud={len(fraud_part)} sample_non_fraud={len(non_fraud_part)}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(dst, index=False)
    print(f"[create_vercel_sample] Wrote PaySim sample to {dst} with {len(df_sample)} rows")


def create_ieee_sample() -> None:
    tx_src = DATA_DIR / IEEE_TX_FILE
    id_src = DATA_DIR / IEEE_ID_FILE
    if not tx_src.exists():
        print(f"[create_vercel_sample] Skipping IEEE samples – transaction source missing at {tx_src}")
        return
    print(f"[create_vercel_sample] Reading IEEE transactions from {tx_src}...")
    tx_full = pd.read_csv(tx_src)
    if tx_full.empty:
        print("[create_vercel_sample] IEEE transaction source is empty; skipping.")
        return

    if "isFraud" in tx_full.columns:
        fraud_tx = tx_full[tx_full["isFraud"] == 1]
        non_fraud_tx = tx_full[tx_full["isFraud"] == 0]
        n_fraud = min(len(fraud_tx), SAMPLE_ROWS // 4) or min(len(fraud_tx), 1000)
        n_non_fraud = min(len(non_fraud_tx), SAMPLE_ROWS - n_fraud)
        fraud_part = fraud_tx.head(n_fraud)
        non_fraud_part = non_fraud_tx.head(n_non_fraud)
        tx_df = pd.concat([fraud_part, non_fraud_part], ignore_index=True)
        print(
            f"[create_vercel_sample] IEEE tx rows={len(tx_full)} fraud_total={len(fraud_tx)} "
            f"sample_rows={len(tx_df)} sample_fraud={len(fraud_part)} sample_non_fraud={len(non_fraud_part)}"
        )
    else:
        print("[create_vercel_sample] IEEE tx has no isFraud column; using head-only sample.")
        tx_df = tx_full.head(SAMPLE_ROWS)

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

