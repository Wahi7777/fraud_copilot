from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterable
import os

import pandas as pd

from models import EnrichedTransactionRecord


@dataclass
class DataRepositoryConfig:
    """
    Configuration for loading local CSV datasets.

    `data_dir` may be overridden via environment variables (see below) and
    `max_rows` is primarily intended for tests and local development so that
    we don't need to load entire large CSVs to validate the pipeline.
    """

    data_dir: Path = Path("data")
    max_rows: Optional[int] = None


class DataRepository:
    """
    Data access layer for fraud investigation datasets.

    Responsibilities:
    - Load PaySim and IEEE fraud CSVs from the `data/` directory
    - Standardise them into `EnrichedTransactionRecord` instances
    - Preserve the original fraud label for each transaction
    - Provide a simple interface for downstream services and agents
    """

    def __init__(self, config: DataRepositoryConfig | None = None) -> None:
        effective_config = config or DataRepositoryConfig()
        mode = os.getenv("DATA_MODE", "").strip().lower()
        override_dir = os.getenv("DATA_DIR") or os.getenv("PAYSIM_DATA_DIR")
        data_dir = effective_config.data_dir
        if override_dir:
            data_dir = Path(override_dir)
        elif mode == "sample":
            # Default sample-mode directory for Vercel or constrained deployments.
            data_dir = data_dir / "vercel_sample"
        self.config = DataRepositoryConfig(data_dir=data_dir, max_rows=effective_config.max_rows)

        # Canonical EnrichedTransactionRecord storage and labels
        self._records: dict[str, EnrichedTransactionRecord] = {}
        self._fraud_labels: dict[str, int] = {}

        # Cached pandas DataFrames
        self._paysim_df: pd.DataFrame | None = None
        self._ieee_tx_df: pd.DataFrame | None = None
        self._ieee_id_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def load_from_csvs(cls, config: DataRepositoryConfig | None = None) -> "DataRepository":
        """
        Build a repository instance from the local CSV datasets.
        """
        repo = cls(config)
        repo.load_paysim()
        repo.load_ieee_transactions()
        return repo

    # ------------------------------------------------------------------ #
    # Public query interface
    # ------------------------------------------------------------------ #
    def get_transaction(self, transaction_id: str) -> Optional[EnrichedTransactionRecord]:
        """
        Return a single enriched transaction by canonical transaction id, if present.
        """
        return self._records.get(transaction_id)

    def get_label(self, transaction_id: str) -> Optional[int]:
        """Return the underlying fraud label (0/1) for a transaction, if known."""
        return self._fraud_labels.get(transaction_id)

    def get_account_history(self, account_id: str, limit: int = 50) -> list[EnrichedTransactionRecord]:
        """
        Return recent transactions for a given account, ordered by timestamp descending.
        """
        records = [r for r in self._records.values() if r.account_id == account_id]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def get_recent_transactions(self, account_id: str, window_steps: int) -> list[EnrichedTransactionRecord]:
        """
        Return transactions for the given account within a synthetic time window.

        For PaySim data, `window_steps` is interpreted as hours. We approximate
        this by using the normalised `timestamp` field and selecting records
        whose timestamp falls within the last `window_steps` hours relative to
        the most recent transaction for that account.
        """
        history = self.get_account_history(account_id, limit=10_000)
        if not history:
            return []

        latest_ts = history[0].timestamp
        window_start = latest_ts - timedelta(hours=window_steps)
        return [r for r in history if r.timestamp >= window_start]

    def get_alert_queue(self, limit: int = 25) -> list[EnrichedTransactionRecord]:
        """
        Return a list of transactions suitable for seeding the alert queue.

        Records are ordered by fraud label (1 before 0) and then by timestamp
        descending to surface the most recent risky activity.
        """
        items: list[tuple[EnrichedTransactionRecord, int]] = []
        for tx_id, record in self._records.items():
            label = self._fraud_labels.get(tx_id, 0)
            items.append((record, label))

        items.sort(key=lambda pair: (pair[1], pair[0].timestamp), reverse=True)
        return [r for r, _ in items[:limit]]

    def iter_records_with_labels(
        self, limit: Optional[int] = None
    ) -> Iterable[tuple[EnrichedTransactionRecord, Optional[int]]]:
        """
        Iterate over enriched records paired with their fraud label.
        """
        count = 0
        for tx_id, record in self._records.items():
            yield record, self._fraud_labels.get(tx_id)
            count += 1
            if limit is not None and count >= limit:
                break

    # ------------------------------------------------------------------ #
    # Dataset loaders with pandas
    # ------------------------------------------------------------------ #
    def load_paysim(self) -> pd.DataFrame:
        """
        Load and standardise the PaySim dataset.

        Returns a pandas DataFrame with at least the following columns:
        - transaction_id
        - account_id
        - beneficiary_id
        - amount
        - transaction_type
        - timestamp
        - is_fraud
        - dataset
        """
        if self._paysim_df is not None:
            return self._paysim_df

        path = self.config.data_dir / "PS_20174392719_1491204439457_log.csv"
        if not path.exists():
            self._paysim_df = pd.DataFrame()
            print(f"[DataRepository] PaySim file not found at {path}, returning empty DataFrame.")
            return self._paysim_df

        kwargs: dict = {}
        if self.config.max_rows is not None:
            kwargs["nrows"] = self.config.max_rows

        raw = pd.read_csv(path, **kwargs)

        base_dt = datetime(2018, 1, 1)

        records = []
        for idx, row in raw.iterrows():
            tx_id = f"PAYSIM-{row.get('nameOrig', '')}-{idx}"

            step_value = row.get("step", 0)
            try:
                step_int = int(step_value)
            except (TypeError, ValueError):
                step_int = 0
            timestamp = base_dt + timedelta(hours=step_int)

            amount = float(row.get("amount", 0.0))
            tx_type = str(row.get("type"))
            account_id = str(row.get("nameOrig"))
            beneficiary_id = str(row.get("nameDest"))
            fraud_label = int(row.get("isFraud", 0))

            records.append(
                {
                    "transaction_id": tx_id,
                    "account_id": account_id,
                    "beneficiary_id": beneficiary_id,
                    "amount": amount,
                    "transaction_type": tx_type,
                    "timestamp": timestamp,
                    "is_fraud": fraud_label,
                    "dataset": "paysim",
                }
            )

            etr = EnrichedTransactionRecord(
                transaction_id=tx_id,
                account_id=account_id,
                beneficiary_id=beneficiary_id,
                amount=amount,
                transaction_type=tx_type,
                timestamp=timestamp,
                device_id=None,
                country=None,
                account_age_days=None,
                transaction_velocity_10min=None,
                merchant_category=None,
                device_risk_score=None,
            )
            self._records[tx_id] = etr
            self._fraud_labels[tx_id] = fraud_label

        self._paysim_df = pd.DataFrame.from_records(records)
        return self._paysim_df

    def load_ieee_identity(self) -> pd.DataFrame:
        """
        Load the IEEE identity dataset (raw, no standardisation applied).
        """
        if self._ieee_id_df is not None:
            return self._ieee_id_df

        path = self.config.data_dir / "train_identity.csv"
        if not path.exists():
            self._ieee_id_df = pd.DataFrame()
            print(f"[DataRepository] IEEE identity file not found at {path}, returning empty DataFrame.")
            return self._ieee_id_df

        kwargs: dict = {}
        if self.config.max_rows is not None:
            kwargs["nrows"] = self.config.max_rows

        self._ieee_id_df = pd.read_csv(path, **kwargs)
        return self._ieee_id_df

    def load_ieee_transactions(self) -> pd.DataFrame:
        """
        Load and standardise the IEEE-CIS fraud transaction dataset, joined with identity.

        Returns a pandas DataFrame with the same canonical columns as PaySim:
        - transaction_id
        - account_id
        - beneficiary_id
        - amount
        - transaction_type
        - timestamp
        - is_fraud
        - dataset
        plus optional enrichment fields `device_id` and `country`.
        """
        if self._ieee_tx_df is not None:
            return self._ieee_tx_df

        tx_path = self.config.data_dir / "train_transaction.csv"
        if not tx_path.exists():
            self._ieee_tx_df = pd.DataFrame()
            print(f"[DataRepository] IEEE transaction file not found at {tx_path}, returning empty DataFrame.")
            return self._ieee_tx_df

        kwargs: dict = {}
        if self.config.max_rows is not None:
            kwargs["nrows"] = self.config.max_rows

        tx_df = pd.read_csv(tx_path, **kwargs)

        id_path = self.config.data_dir / "train_identity.csv"
        if id_path.exists():
            id_df = self.load_ieee_identity()
            merged = tx_df.merge(id_df, on="TransactionID", how="left", suffixes=("", "_id"))
        else:
            merged = tx_df

        base_dt = datetime(2017, 1, 1)

        records = []
        for _, row in merged.iterrows():
            raw_tx_id = row.get("TransactionID")
            if pd.isna(raw_tx_id):
                continue
            tx_id = f"IEEE-{int(raw_tx_id)}"

            amount = float(row.get("TransactionAmt", 0.0))
            product_cd = row.get("ProductCD")
            card1 = row.get("card1")
            addr1 = row.get("addr1")

            dt_value = row.get("TransactionDT", 0)
            try:
                dt_int = int(dt_value)
            except (TypeError, ValueError):
                dt_int = 0
            timestamp = base_dt + timedelta(seconds=dt_int)

            device_info = row.get("DeviceInfo")
            device_type = row.get("DeviceType")
            id_31 = row.get("id_31")

            device_id = None
            for candidate in (device_info, device_type, id_31):
                if isinstance(candidate, str) and candidate:
                    device_id = candidate
                    break

            country = None
            if not pd.isna(addr1):
                country = f"ADDR-{int(addr1)}"

            account_id = str(card1) if not pd.isna(card1) else "UNKNOWN"
            beneficiary_id = "UNKNOWN" if pd.isna(addr1) else f"BENEF-{int(addr1)}"
            tx_type = str(product_cd) if not pd.isna(product_cd) else "UNKNOWN"
            fraud_label = int(row.get("isFraud", 0))

            records.append(
                {
                    "transaction_id": tx_id,
                    "account_id": account_id,
                    "beneficiary_id": beneficiary_id,
                    "amount": amount,
                    "transaction_type": tx_type,
                    "timestamp": timestamp,
                    "is_fraud": fraud_label,
                    "dataset": "ieee",
                    "device_id": device_id,
                    "country": country,
                }
            )

            etr = EnrichedTransactionRecord(
                transaction_id=tx_id,
                account_id=account_id,
                beneficiary_id=beneficiary_id,
                amount=amount,
                transaction_type=tx_type,
                timestamp=timestamp,
                device_id=device_id,
                country=country,
                account_age_days=None,
                transaction_velocity_10min=None,
                merchant_category=None,
                device_risk_score=None,
            )

            self._records[tx_id] = etr
            self._fraud_labels[tx_id] = fraud_label

        self._ieee_tx_df = pd.DataFrame.from_records(records)
        return self._ieee_tx_df

