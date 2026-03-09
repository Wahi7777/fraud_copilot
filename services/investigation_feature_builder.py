from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from models import EnrichedTransactionRecord
from services.data_repository import DataRepository


@dataclass
class InvestigationContext:
    """
    Lightweight context object for an investigation.

    For now this only wraps the transaction identifier, but can be extended
    later (e.g. with alert metadata) without changing the builder interface.
    """

    transaction_id: str


class InvestigationFeatureBuilder:
    """
    Builds a fully-populated EnrichedTransactionRecord for a given alert.

    The builder:
    - Uses PaySim as the base transaction source
    - Uses IEEE transaction/identity data as enrichment when available
    - Falls back to deterministic synthetic enrichments when direct joins
      are not possible
    All enrichment logic is deterministic and reproducible.
    """

    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository

        # Eagerly load IEEE data once so enrichment stays deterministic
        self._ieee_df = self.repository.load_ieee_transactions()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_for_transaction(self, ctx: InvestigationContext) -> EnrichedTransactionRecord:
        """
        Build an EnrichedTransactionRecord for the given investigation context.
        """
        base = self.repository.get_transaction(ctx.transaction_id)
        if base is None:
            raise ValueError(f"Unknown transaction id: {ctx.transaction_id}")

        enriched = base.copy()

        # Account / beneficiary age – deterministic synthetic values if missing.
        enriched = self._with_age_features(enriched)

        # Velocity features from account history.
        enriched = self._with_velocity_features(enriched)

        # IEEE-based device/geo enrichment if possible.
        enriched = self._with_ieee_enrichment(enriched)

        # Deterministic synthetic signals (IP, email domain risk, geo jump).
        enriched = self._with_synthetic_signals(enriched)

        return enriched

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _with_age_features(self, record: EnrichedTransactionRecord) -> EnrichedTransactionRecord:
        if record.account_age_days is None:
            record.account_age_days = 30 + (abs(hash(record.account_id)) % 3650)

        if record.beneficiary_age_days is None:
            record.beneficiary_age_days = 7 + (abs(hash(record.beneficiary_id)) % 3650)

        return record

    def _with_velocity_features(self, record: EnrichedTransactionRecord) -> EnrichedTransactionRecord:
        history = self.repository.get_account_history(record.account_id, limit=10_000)

        window_10min = record.timestamp - timedelta(minutes=10)
        window_1hr = record.timestamp - timedelta(hours=1)

        tx_10min = [
            h for h in history if window_10min <= h.timestamp < record.timestamp
        ]
        tx_1hr = [
            h for h in history if window_1hr <= h.timestamp < record.timestamp
        ]

        record.transaction_velocity_10min = float(len(tx_10min))
        record.transaction_velocity_1hr = float(len(tx_1hr))

        return record

    def _select_ieee_row_for_account(self, account_id: str) -> Optional[dict]:
        """
        Deterministically select an IEEE row to act as enrichment source for
        a given account id. This is necessary because PaySim and IEEE are
        disjoint datasets in the MVP.
        """
        if self._ieee_df is None or self._ieee_df.empty:
            return None

        idx = abs(hash(account_id)) % len(self._ieee_df)
        row = self._ieee_df.iloc[int(idx)]
        return row.to_dict()

    def _with_ieee_enrichment(self, record: EnrichedTransactionRecord) -> EnrichedTransactionRecord:
        ieee_row = self._select_ieee_row_for_account(record.account_id)
        if ieee_row is None:
            return record

        device_id = ieee_row.get("device_id") or record.device_id
        country = ieee_row.get("country") or record.country

        # Device type heuristic derived from device_id.
        device_type: Optional[str] = record.device_type
        if device_type is None and isinstance(device_id, str):
            lowered = device_id.lower()
            if "iphone" in lowered or "android" in lowered or "mobile" in lowered:
                device_type = "MOBILE"
            elif "ipad" in lowered or "tablet" in lowered:
                device_type = "TABLET"
            else:
                device_type = "DESKTOP"

        record.device_id = device_id
        record.country = country
        record.device_type = device_type

        return record

    def _with_synthetic_signals(self, record: EnrichedTransactionRecord) -> EnrichedTransactionRecord:
        if not record.device_id:
            record.device_id = f"DEV-{abs(hash(record.account_id)) % 100000:05d}"
        if not record.device_type:
            token = str(record.device_id).lower()
            if "mobile" in token or token.startswith("dev-"):
                record.device_type = "MOBILE"
            else:
                record.device_type = "DESKTOP"
        # Deterministic synthetic IP address from transaction id.
        record.ip_address = _synthetic_ip_for_key(record.transaction_id)

        # Device risk score – higher for accounts whose hashed id falls in upper bands
        # and for transactions labelled as fraud where that information is available.
        base_score = (abs(hash(record.account_id)) % 50) / 100.0  # 0.00–0.49

        # Normalise into a slightly riskier range for demo.
        base_score = 0.25 + base_score  # 0.25–0.74
        record.device_risk_score = min(base_score, 0.95)

        # Email domain risk – purely synthetic but deterministic.
        record.email_domain_risk = (abs(hash(record.account_id + record.beneficiary_id)) % 100) / 100.0

        # Geo jump and impossible travel from previous transaction (if any).
        history = self.repository.get_account_history(record.account_id, limit=10_000)
        previous = _previous_transaction(history, record)

        if previous is None or previous.country is None or record.country is None:
            record.geo_distance_jump_km = 0.0
            record.impossible_travel_flag = False
            return record

        # If country changed between subsequent transactions, simulate a fixed
        # long-distance jump; otherwise treat as local.
        if previous.country != record.country:
            record.geo_distance_jump_km = 4200.0

            time_delta_hours = max(
                (record.timestamp - previous.timestamp).total_seconds() / 3600.0, 0.0
            )
            # Impossible travel if jump is > 1000km and time delta < 2 hours.
            record.impossible_travel_flag = time_delta_hours < 2.0
        else:
            record.geo_distance_jump_km = 0.0
            record.impossible_travel_flag = False

        return record


def _synthetic_ip_for_key(key: str) -> str:
    """
    Deterministically map an arbitrary key onto a synthetic IPv4 address.
    """
    h = abs(hash(key))
    octets = []
    for shift in (0, 8, 16, 24):
        part = (h >> shift) & 0xFF
        if part == 0:
            part = 1
        if part == 255:
            part = 254
        octets.append(part)
    return ".".join(str(o) for o in octets)


def _previous_transaction(
    history: list[EnrichedTransactionRecord], current: EnrichedTransactionRecord
) -> Optional[EnrichedTransactionRecord]:
    """
    Return the chronologically previous transaction in history before `current`.
    """
    candidates = [h for h in history if h.timestamp < current.timestamp]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.timestamp, reverse=True)
    return candidates[0]

