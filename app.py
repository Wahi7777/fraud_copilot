import os

from datetime import datetime
from functools import lru_cache
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import EnrichedTransactionRecord
from services.data_repository import DataRepository, DataRepositoryConfig
from services.llm_provider import get_llm_runtime_info


# Load environment variables (including OPENAI_API_KEY) from .env if present
load_dotenv()
logger = logging.getLogger("fraud_api")


def _risk_from_tx(tx: EnrichedTransactionRecord) -> float:
    base = tx.device_risk_score or 0.3
    velocity = (tx.transaction_velocity_10min or 0.0) / 10.0
    return float(min(base + velocity, 0.99))


def _alert_type_from_tx(tx: EnrichedTransactionRecord) -> str:
    return "High Velocity Transfer" if (tx.transaction_velocity_10min or 0.0) >= 3 else "Transaction Anomaly"


def _normalize_severity(value: str | None) -> str:
    token = str(value or "").lower()
    if token in {"critical", "high"}:
        return "high"
    if token == "medium":
        return "medium"
    return "low"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_location(country: Any) -> str:
    token = str(country or "").strip()
    if not token:
        return "Unknown"
    if token.upper().startswith("ADDR-"):
        return "Unknown"
    return token


def _build_modal_signals(base_tx: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Normalize raw transaction/evidence data into dashboard signal cards.
    """
    signals: Dict[str, Dict[str, str]] = {}

    velocity_10m = _to_float(base_tx.get("transaction_velocity_10min"))
    velocity_1h = _to_float(base_tx.get("transaction_velocity_1hr"))
    if velocity_10m > 0 or velocity_1h > 0:
        sev = "high" if velocity_10m >= 3 else "medium" if velocity_10m >= 1 else "low"
        signals["velocity"] = {
            "label": "Velocity",
            "value": f"{velocity_10m:.0f} tx / 10m",
            "detail": f"{velocity_1h:.0f} tx in last hour",
            "severity": sev,
        }

    amount = _to_float(base_tx.get("amount"))
    balance_after = _to_float(base_tx.get("balance_after"), default=-1.0)
    balance_before = _to_float(base_tx.get("balance_before"), default=-1.0)
    inferred_balance = balance_before if balance_before > 0 else (amount + balance_after if balance_after >= 0 else -1.0)
    utilization = (amount / inferred_balance) if inferred_balance > 0 else None
    if amount > 0:
        if utilization is not None and utilization >= 0.6:
            drain_sev = "high" if utilization >= 0.9 else "medium"
            signals["balance_drain"] = {
                "label": "Balance Drain",
                "value": f"${amount:,.2f} ({int(utilization * 100)}% of account balance)",
                "detail": "High balance utilization / possible drain pattern",
                "severity": drain_sev,
            }
        else:
            amount_sev = "high" if amount >= 5000 else "medium" if amount >= 1000 else "low"
            signals["transfer_amount"] = {
                "label": "Transfer Amount",
                "value": f"${amount:,.2f}",
                "detail": "Outbound transfer amount under review",
                "severity": amount_sev,
            }

    device_risk = _to_float(base_tx.get("device_risk_score"))
    if device_risk > 0:
        sev = "high" if device_risk >= 0.75 else "medium" if device_risk >= 0.5 else "low"
        signals["device_reuse"] = {
            "label": "Device Risk",
            "value": f"{int(device_risk * 100)}%",
            "detail": "Device risk score / possible reuse pattern",
            "severity": sev,
        }

    geo_jump = _to_float(base_tx.get("geo_distance_jump_km"))
    impossible = bool(base_tx.get("impossible_travel_flag"))
    if impossible or geo_jump > 0:
        sev = "high" if impossible or geo_jump >= 1500 else "medium" if geo_jump >= 300 else "low"
        signals["geo_anomaly"] = {
            "label": "Geo Anomaly",
            "value": "Impossible travel" if impossible else f"{geo_jump:.0f} km",
            "detail": "Geo distance anomaly from recent activity",
            "severity": sev,
        }

    for ev in evidence:
        code = str(ev.get("signal_code") or "").upper()
        summary = str(ev.get("summary") or "").strip() or "Detected by investigation"
        severity = _normalize_severity(ev.get("severity"))

        if any(token in code for token in ["VELOCITY", "SPIKE"]):
            signals["velocity"] = {
                "label": "Velocity",
                "value": signals.get("velocity", {}).get("value", "Anomalous velocity"),
                "detail": summary,
                "severity": severity,
            }
        elif any(token in code for token in ["BALANCE", "DRAIN", "AMOUNT"]):
            signals["balance_drain"] = {
                "label": "Balance Drain",
                "value": signals.get("balance_drain", {}).get("value", "Balance movement anomaly"),
                "detail": summary,
                "severity": severity,
            }
        elif any(token in code for token in ["DEVICE", "FINGERPRINT"]):
            signals["device_reuse"] = {
                "label": "Device Risk",
                "value": signals.get("device_reuse", {}).get("value", "Device risk"),
                "detail": summary,
                "severity": severity,
            }
        elif any(token in code for token in ["GEO", "TRAVEL", "LOCATION"]):
            signals["geo_anomaly"] = {
                "label": "Geo Anomaly",
                "value": signals.get("geo_anomaly", {}).get("value", "Geo anomaly"),
                "detail": summary,
                "severity": severity,
            }

    logger.info("[INVESTIGATE] raw signals evidence count=%d", len(evidence))
    logger.info("[INVESTIGATE] normalized signals keys=%s", list(signals.keys()))
    return signals


def _network_summary(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    codes = {str(e.get("signal_code") or "").upper() for e in evidence}
    linked = "LINKED_FRAUD_ACCOUNT" in codes
    suspected = linked or any(code in {"MULE_PATTERN", "RAPID_BENEFICIARY_CHANGE", "SHARED_DEVICE_NETWORK"} for code in codes)
    if linked:
        label = "High Network Risk"
        link_status = "Confirmed link"
    elif suspected:
        label = "Suspected Network Risk"
        link_status = "Suspected link"
    else:
        label = "Low Network Risk"
        link_status = "No known fraud link"
    return {"fraud_link": linked, "suspected_network_risk": suspected and not linked, "network_risk_label": label, "fraud_link_status": link_status}


def _derive_typology(signals: Dict[str, Dict[str, str]], network: Dict[str, Any]) -> str:
    high_velocity = signals.get("velocity", {}).get("severity") == "high"
    high_device = signals.get("device_reuse", {}).get("severity") in {"high", "medium"}
    geo_flag = "geo_anomaly" in signals
    if network.get("fraud_link"):
        return "Potential Mule Transfer"
    if high_device and geo_flag:
        return "Possible Account Takeover"
    if high_velocity:
        return "Suspicious Transfer Pattern"
    if "transfer_amount" in signals or "balance_drain" in signals:
        return "Transaction Anomaly"
    return "Suspicious Transaction Activity"


def _risk_drivers(signals: Dict[str, Dict[str, str]], network: Dict[str, Any], risk_score: float) -> List[str]:
    drivers: List[str] = []
    for key, label in [
        ("velocity", "Unusual transaction velocity"),
        ("balance_drain", "High balance utilization / drain behavior"),
        ("transfer_amount", "Elevated outbound transfer amount"),
        ("device_reuse", "Device risk / reuse signal"),
        ("geo_anomaly", "Geolocation anomaly"),
    ]:
        signal = signals.get(key)
        if signal and signal.get("severity") in {"high", "medium"}:
            drivers.append(label)
    if network.get("fraud_link"):
        drivers.append("Confirmed link to known risky beneficiary")
    elif network.get("suspected_network_risk"):
        drivers.append("Beneficiary shows suspected network-risk pattern")
    if not drivers and risk_score >= 0.8:
        drivers.append("Agent consensus indicates elevated fraud risk requiring analyst review")
    return drivers[:4]


def _severity_factor(level: str | None) -> float:
    token = str(level or "").lower()
    if token == "high":
        return 1.0
    if token == "medium":
        return 0.6
    if token == "low":
        return 0.3
    return 0.0


def _has_proxy_risk(evidence: List[Dict[str, Any]]) -> bool:
    for ev in evidence:
        text = f"{ev.get('summary', '')} {ev.get('details', '')}".lower()
        if any(k in text for k in ["proxy", "vpn", "tor", "anonym", "masked ip"]):
            return True
    return False


def _has_repeated_critical(evidence: List[Dict[str, Any]]) -> bool:
    critical_count = 0
    for ev in evidence:
        if str(ev.get("severity") or "").lower() == "critical":
            critical_count += 1
    return critical_count >= 2


def _compute_final_policy(
    *,
    signals: Dict[str, Dict[str, str]],
    network: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    llm_risk_score: float,
) -> Dict[str, Any]:
    """
    Deterministic final scoring and action policy.

    Weighted additive model with bounded signal contributions:
    - transaction_anomaly: max 0.16
    - device_risk: max 0.16
    - ip_proxy_risk: max 0.14
    - velocity: max 0.16
    - balance_utilization: max 0.14
    - beneficiary_risk: max 0.24
    """
    velocity = 0.16 * _severity_factor(signals.get("velocity", {}).get("severity"))
    device_risk = 0.16 * _severity_factor(signals.get("device_reuse", {}).get("severity"))
    transaction_anomaly = 0.16 * _severity_factor(
        (signals.get("transfer_amount") or signals.get("balance_drain") or {}).get("severity")
    )
    balance_utilization = 0.14 * _severity_factor(signals.get("balance_drain", {}).get("severity"))
    ip_proxy_risk = 0.14 if _has_proxy_risk(evidence) else 0.03
    if network.get("fraud_link"):
        beneficiary_risk = 0.24
    elif network.get("suspected_network_risk"):
        beneficiary_risk = 0.12
    else:
        beneficiary_risk = 0.02

    # Keep some influence from model reasoning, but bounded so policy stays controllable.
    llm_adjustment = min(max(llm_risk_score, 0.0), 1.0) * 0.06
    base_prior = 0.06
    contributions = {
        "transaction_anomaly": round(transaction_anomaly, 3),
        "device_risk": round(device_risk, 3),
        "ip_proxy_risk": round(ip_proxy_risk, 3),
        "velocity": round(velocity, 3),
        "balance_utilization": round(balance_utilization, 3),
        "beneficiary_risk": round(beneficiary_risk, 3),
        "llm_adjustment": round(llm_adjustment, 3),
        "base_prior": round(base_prior, 3),
    }
    final_score = min(round(sum(contributions.values()), 3), 0.99)

    if final_score < 0.30:
        final_band = "low"
    elif final_score < 0.65:
        final_band = "medium"
    elif final_score < 0.85:
        final_band = "high"
    else:
        final_band = "critical"

    high_like_count = sum(
        1
        for key in ["transaction_anomaly", "device_risk", "velocity", "balance_utilization"]
        if contributions.get(key, 0.0) >= 0.11
    )
    strong_ato = (
        _severity_factor(signals.get("device_reuse", {}).get("severity")) >= 1.0
        and _severity_factor(signals.get("geo_anomaly", {}).get("severity")) >= 1.0
        and contributions["ip_proxy_risk"] >= 0.14
    )
    hard_stop = bool(
        network.get("fraud_link")
        or strong_ato
        or _has_repeated_critical(evidence)
        or (high_like_count >= 3 and final_score >= 0.90)
    )
    hard_stop_reasons: List[str] = []
    if network.get("fraud_link"):
        hard_stop_reasons.append("confirmed_fraud_link")
    if strong_ato:
        hard_stop_reasons.append("strong_ato_pattern")
    if _has_repeated_critical(evidence):
        hard_stop_reasons.append("repeated_critical_behavior")
    if high_like_count >= 3 and final_score >= 0.90:
        hard_stop_reasons.append("multiple_high_severity_signals")

    if final_band == "critical" and hard_stop:
        action = "freeze_account"
        recommendation = "Freeze account and escalate immediately due to confirmed critical fraud indicators."
    elif final_band in {"high", "critical"}:
        action = "decline_transaction"
        recommendation = "Decline transaction and escalate for fraud analyst review."
    elif final_band == "medium":
        action = "hold_and_review"
        recommendation = "Hold transaction and perform step-up verification with analyst review."
    else:
        action = "mark_false_positive" if final_score < 0.20 else "close_case"
        recommendation = "Close alert with monitoring; no immediate blocking action."

    logger.warning(
        "[POLICY DEBUG] contributions=%s final_score=%.3f final_band=%s final_action=%s hard_stop=%s hard_stop_reasons=%s",
        contributions,
        final_score,
        final_band,
        action,
        hard_stop,
        hard_stop_reasons,
    )
    return {
        "final_score": final_score,
        "final_band": final_band,
        "final_action": action,
        "final_recommendation": recommendation,
        "hard_stop_triggered": hard_stop,
        "hard_stop_reasons": hard_stop_reasons,
        "contributions": contributions,
    }


def _map_decision_to_queue_status(decision: Dict[str, Any] | None) -> str:
    token = str(
        (decision or {}).get("recommended_action")
        or (decision or {}).get("action")
        or (decision or {}).get("recommendation")
        or ""
    ).lower()
    if "decline_transaction" in token:
        return "Decline"
    if "hold_and_review" in token or "step_up_auth" in token:
        return "Hold"
    if "freeze_account" in token:
        return "Freeze"
    if "escalate_case" in token:
        return "Escalate"
    if "mark_false_positive" in token:
        return "False Positive"
    if "close_case" in token:
        return "Closed"
    return "Closed"


@lru_cache(maxsize=1)
def _get_investigation_store() -> Dict[str, Dict[str, Any]]:
    """
    In-memory store of completed investigations keyed by alert_id.
    """
    return {}


@lru_cache(maxsize=1)
def _get_investigation_progress_store() -> Dict[str, Dict[str, Any]]:
    return {}


@lru_cache(maxsize=1)
def _get_progress_lock() -> threading.Lock:
    return threading.Lock()


@lru_cache(maxsize=1)
def _get_repository_singleton() -> DataRepository:
    """
    Lightweight shared repository for dashboard/read APIs.

    Loads only PaySim during bootstrap so /alerts and /metrics remain fast.
    IEEE data is loaded lazily later when an investigation needs enrichment.
    """
    t0 = time.perf_counter()
    # Use a bounded sample for dashboard APIs so startup remains fast in MVP.
    sample_rows = int(os.getenv("DASHBOARD_SAMPLE_ROWS", "200000"))
    repo = DataRepository(DataRepositoryConfig(max_rows=sample_rows))
    repo.load_paysim()
    elapsed = time.perf_counter() - t0
    logger.info("repository_init_seconds=%.3f", elapsed)
    return repo


@lru_cache(maxsize=1)
def _get_alert_catalog() -> List[Dict[str, Any]]:
    """
    Precomputed alert payloads for fast dashboard reads.
    """
    repo = _get_repository_singleton()
    records = repo.get_alert_queue(limit=5000)

    catalog: List[Dict[str, Any]] = []
    for tx in records:
        risk_score = _risk_from_tx(tx)
        if risk_score >= 0.7:
            risk_level, status = "HIGH", "Open"
        elif risk_score >= 0.4:
            risk_level, status = "MEDIUM", "Reviewing"
        else:
            risk_level, status = "LOW", "Pending"

        catalog.append(
            {
                "alert_id": tx.transaction_id,
                "transaction_id": tx.transaction_id,
                "account_id": tx.account_id,
                "beneficiary_id": tx.beneficiary_id,
                "alert_type": _alert_type_from_tx(tx),
                "amount": tx.amount,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "timestamp": tx.timestamp,
                "status": status,
            }
        )
    return catalog


@lru_cache(maxsize=1)
def _get_metrics_summary() -> Dict[str, Any]:
    repo = _get_repository_singleton()
    df = repo.load_paysim()
    total = int(len(df))
    flagged = int(df["is_fraud"].sum()) if not df.empty else 0

    high_risk_alerts = flagged
    escalated_cases = max(1, flagged // 10) if flagged else 0
    open_investigations = min(total, max(flagged * 2, 10))

    return {
        "flagged_transactions": flagged,
        "high_risk_alerts": high_risk_alerts,
        "open_investigations": open_investigations,
        "escalated_cases": escalated_cases,
        "false_positive_rate": 0.15,
        "avg_resolution_time": 14.0,
    }


@lru_cache(maxsize=1)
def _get_pipeline_singleton():
    """
    Heavy path initializer (CrewAI + pipeline). Only used for real investigations.
    """
    t0 = time.perf_counter()
    from services.investigation_pipeline import InvestigationPipeline

    pipeline = InvestigationPipeline(repository=_get_repository_singleton())
    elapsed = time.perf_counter() - t0
    logger.info("pipeline_init_seconds=%.3f", elapsed)
    return pipeline


def create_app() -> FastAPI:
    """
    Application factory for the Fraud Investigation Co-Pilot backend.

    This keeps the FastAPI instance construction isolated so it can be reused
    by tests and deployment runners (uvicorn, gunicorn, etc.).
    """
    app = FastAPI(
        title="Agentic Fraud Investigation Co-Pilot MVP",
        version="0.1.0",
        description=(
            "Backend API for an agentic fraud investigation co-pilot. "
            "Implements the investigation pipeline, CrewAI agents, and "
            "supporting services as defined in the MVP PRD."
        ),
    )

    # CORS configuration – relaxed for MVP/demo, can be tightened later
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5500",
            "http://127.0.0.1:5500",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health_check() -> dict:
        """
        Lightweight health endpoint used by tests and orchestrators.
        """
        return {"status": "ok"}

    # ------------------------------------------------------------------ #
    # Lightweight dependency providers
    # ------------------------------------------------------------------ #

    def get_repository() -> DataRepository:
        return _get_repository_singleton()

    # ------------------------------------------------------------------ #
    # Pydantic request/response models
    # ------------------------------------------------------------------ #

    class InvestigateRequest(BaseModel):
        transaction_id: str = Field(..., description="Transaction identifier to investigate.")

    class InvestigationData(BaseModel):
        alert_id: Optional[str] = None
        customer_id: Optional[str] = None
        account_id: Optional[str] = None
        timestamp: Optional[datetime] = None
        risk_score: Optional[float] = None
        status: str
        investigation: Optional[dict]
        evidence: List[dict]
        decision: Optional[dict]
        case_summary: Optional[dict] = None
        device_identity: Optional[dict] = None
        signals: Optional[Dict[str, Dict[str, str]]] = None
        supporting_transactions: Optional[List[dict]] = None
        beneficiary_analysis: Optional[dict] = None
        agent_trace: Optional[List[Dict[str, str]]] = None
        llm_provider_used: Optional[str] = None
        model_used: Optional[str] = None
        investigation_status: Optional[str] = None
        investigation_completed_at: Optional[datetime] = None
        final_risk_score: Optional[float] = None
        final_recommendation: Optional[str] = None
        queue_status: Optional[str] = None

    class InvestigationEnvelope(BaseModel):
        ok: bool
        data: Optional[InvestigationData] = None
        error: Optional[str] = None
        details: Optional[str] = None

    class InvestigationProgressEnvelope(BaseModel):
        ok: bool
        status: str
        events: List[Dict[str, Any]] = Field(default_factory=list)
        data: Optional[InvestigationData] = None
        error: Optional[str] = None
        details: Optional[str] = None

    class AlertItem(BaseModel):
        alert_id: str
        transaction_id: str
        account_id: str
        beneficiary_id: str
        alert_type: str
        amount: float
        risk_score: float
        risk_level: str
        timestamp: datetime
        status: str
        investigation_status: str = "new"
        investigation_completed_at: Optional[datetime] = None
        investigation_summary: Optional[str] = None
        final_risk_score: Optional[float] = None
        final_recommendation: Optional[str] = None
        has_cached_investigation: bool = False
        last_investigation_id: Optional[str] = None
        queue_status: Optional[str] = None

    class AlertsResponse(BaseModel):
        items: List[AlertItem]

    class MetricsResponse(BaseModel):
        flagged_transactions: int
        high_risk_alerts: int
        open_investigations: int
        escalated_cases: int
        false_positive_rate: float
        avg_resolution_time: float  # minutes

    # ------------------------------------------------------------------ #
    # /investigate
    # ------------------------------------------------------------------ #

    @app.post("/investigate", response_model=InvestigationEnvelope, tags=["investigation"])
    def investigate(
        body: InvestigateRequest,
        repo: DataRepository = Depends(get_repository),
    ) -> InvestigationEnvelope:
        """
        Run the full investigation pipeline for a given transaction identifier.

        For missing transactions we short-circuit with 404 and avoid
        constructing or executing the CrewAI-based pipeline entirely.
        """
        if repo.get_transaction(body.transaction_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction {body.transaction_id} not found in repository.",
            )

        result = _run_investigation_for_alert(body.transaction_id)
        return InvestigationEnvelope(**result)

    @app.get("/investigate/{transaction_id}", response_model=InvestigationEnvelope, tags=["investigation"])
    def investigate_by_id(
        transaction_id: str,
        repo: DataRepository = Depends(get_repository),
    ) -> InvestigationEnvelope:
        """
        Fetch investigation result for a transaction id via path parameter.
        Mirrors POST /investigate behaviour for frontend convenience.
        """
        if repo.get_transaction(transaction_id) is None:
            return InvestigationEnvelope(
                ok=False,
                error="Transaction not found",
                details=f"Transaction {transaction_id} not found in repository.",
            )

        result = _run_investigation_for_alert(transaction_id)
        return InvestigationEnvelope(**result)

    @app.post("/investigate/{alert_id}", response_model=InvestigationEnvelope, tags=["investigation"])
    def investigate_by_alert_id(
        alert_id: str,
        repo: DataRepository = Depends(get_repository),
    ) -> InvestigationEnvelope:
        """
        Generate investigation modal payload lazily when user clicks Investigate.
        """
        if repo.get_transaction(alert_id) is None:
            return InvestigationEnvelope(
                ok=False,
                error="Transaction not found",
                details=f"Transaction {alert_id} not found in repository.",
            )

        result = _run_investigation_for_alert(alert_id)
        return InvestigationEnvelope(**result)

    @app.post("/investigate/{alert_id}/start", response_model=InvestigationProgressEnvelope, tags=["investigation"])
    def start_investigation_progress(
        alert_id: str,
        repo: DataRepository = Depends(get_repository),
    ) -> InvestigationProgressEnvelope:
        """
        Start investigation in background and expose progressive agent events.
        """
        if repo.get_transaction(alert_id) is None:
            return InvestigationProgressEnvelope(
                ok=False,
                status="failed",
                error="Transaction not found",
                details=f"Transaction {alert_id} not found in repository.",
            )

        store = _get_investigation_progress_store()
        lock = _get_progress_lock()
        with lock:
            current = store.get(alert_id)
            if current and current.get("status") == "running":
                return InvestigationProgressEnvelope(
                    ok=True,
                    status="running",
                    events=current.get("events", []),
                    data=current.get("data"),
                )
            store[alert_id] = {
                "status": "running",
                "events": [],
                "data": None,
                "error": None,
                "details": None,
            }

        def emit_progress(event: Dict[str, Any]) -> None:
            with lock:
                entry = store.get(alert_id)
                if not entry:
                    return
                entry.setdefault("events", []).append(jsonable_encoder(event))

        def run_job() -> None:
            result = _run_investigation_for_alert(alert_id, progress_callback=emit_progress)
            with lock:
                entry = store.get(alert_id)
                if not entry:
                    return
                if result.get("ok"):
                    entry["status"] = "completed"
                    entry["data"] = result.get("data")
                else:
                    entry["status"] = "failed"
                    entry["error"] = result.get("error")
                    entry["details"] = result.get("details")

        threading.Thread(target=run_job, daemon=True).start()
        return InvestigationProgressEnvelope(ok=True, status="running", events=[])

    @app.get("/investigate/{alert_id}/progress", response_model=InvestigationProgressEnvelope, tags=["investigation"])
    def get_investigation_progress(alert_id: str) -> InvestigationProgressEnvelope:
        store = _get_investigation_progress_store()
        entry = store.get(alert_id)
        if not entry:
            return InvestigationProgressEnvelope(
                ok=False,
                status="not_found",
                error="Investigation progress not found",
                details=f"No active investigation progress for alert {alert_id}.",
            )
        return InvestigationProgressEnvelope(
            ok=entry.get("status") in {"running", "completed"},
            status=entry.get("status", "running"),
            events=entry.get("events", []),
            data=entry.get("data"),
            error=entry.get("error"),
            details=entry.get("details"),
        )

    @app.get("/investigation/{alert_id}", response_model=InvestigationEnvelope, tags=["investigation"])
    def get_completed_investigation(alert_id: str) -> InvestigationEnvelope:
        """
        Return the latest stored completed investigation for an alert, if available.
        """
        store = _get_investigation_store()
        entry = store.get(alert_id)
        if not entry:
            return InvestigationEnvelope(
                ok=False,
                error="Investigation not found",
                details=f"Alert {alert_id} has not been investigated yet.",
            )
        return InvestigationEnvelope(ok=True, data=entry.get("data"))

    # ------------------------------------------------------------------ #
    # /alerts
    # ------------------------------------------------------------------ #

    @app.get("/alerts", response_model=AlertsResponse, tags=["dashboard"])
    def get_alerts(
        limit: int = Query(25, ge=1, le=200),
        min_risk_score: Optional[float] = Query(None, ge=0.0, le=0.99),
        max_risk_score: Optional[float] = Query(None, ge=0.0, le=0.99),
        sort_by: str = Query("risk_score", pattern="^(risk_score|timestamp)$"),
        sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
    ) -> AlertsResponse:
        """
        Return a queue of flagged alerts suitable for populating the dashboard.
        """
        t0 = time.perf_counter()
        catalog = _get_alert_catalog()
        items: List[AlertItem] = []
        for row in catalog:
            store_entry = _get_investigation_store().get(row["alert_id"])
            risk_score = float(row["risk_score"])
            queue_status = row["status"]
            investigation_status = "new"
            investigation_completed_at = None
            investigation_summary = None
            final_risk_score = None
            final_recommendation = None
            has_cached_investigation = False
            last_investigation_id = None

            if store_entry:
                data = store_entry.get("data") or {}
                decision = data.get("decision") or {}
                risk_score = float(data.get("risk_score", risk_score))
                queue_status = _map_decision_to_queue_status(decision)
                investigation_status = "investigated"
                investigation_completed_at = store_entry.get("investigation_completed_at")
                investigation_summary = (
                    (data.get("case_summary") or {}).get("typology")
                    or decision.get("decision_rationale")
                )
                final_risk_score = risk_score
                final_recommendation = decision.get("recommendation")
                has_cached_investigation = True
                last_investigation_id = (data.get("investigation") or {}).get("investigation_id")

            if min_risk_score is not None and risk_score < min_risk_score:
                continue
            if max_risk_score is not None and risk_score > max_risk_score:
                continue
            row_payload = dict(row)
            row_payload["risk_score"] = risk_score
            row_payload["risk_level"] = "HIGH" if risk_score >= 0.7 else "MEDIUM" if risk_score >= 0.4 else "LOW"
            row_payload["status"] = queue_status
            row_payload["queue_status"] = queue_status
            if investigation_completed_at:
                row_payload["timestamp"] = investigation_completed_at
            row_payload["investigation_status"] = investigation_status
            row_payload["investigation_completed_at"] = investigation_completed_at
            row_payload["investigation_summary"] = investigation_summary
            row_payload["final_risk_score"] = final_risk_score
            row_payload["final_recommendation"] = final_recommendation
            row_payload["has_cached_investigation"] = has_cached_investigation
            row_payload["last_investigation_id"] = last_investigation_id
            items.append(AlertItem(**row_payload))

        # Sort and enforce final limit
        reverse = sort_dir == "desc"
        if sort_by == "risk_score":
            items.sort(key=lambda a: a.risk_score, reverse=reverse)
        else:
            items.sort(key=lambda a: a.timestamp, reverse=reverse)

        payload = AlertsResponse(items=items[:limit])
        elapsed = time.perf_counter() - t0
        logger.info("alerts_route_seconds=%.3f items=%d", elapsed, len(payload.items))
        return payload

    # ------------------------------------------------------------------ #
    # /metrics
    # ------------------------------------------------------------------ #

    @app.get("/metrics", response_model=MetricsResponse, tags=["dashboard"])
    def get_metrics() -> MetricsResponse:
        """
        Return KPI-style metrics used to populate dashboard summary cards.
        """
        t0 = time.perf_counter()
        summary = _get_metrics_summary()
        payload = MetricsResponse(**summary)
        elapsed = time.perf_counter() - t0
        logger.info("metrics_route_seconds=%.3f", elapsed)
        return payload

    def _augment_investigation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shape investigation output into dashboard-friendly modal sections.
        """
        inv = payload.get("investigation") or {}
        base_tx = inv.get("base_transaction") or {}
        decision = payload.get("decision") or {}
        evidence = payload.get("evidence") or []
        signals = _build_modal_signals(base_tx, evidence)
        network = _network_summary(evidence)
        llm_score = _to_float(decision.get("risk_score", inv.get("risk_score")))
        policy = _compute_final_policy(
            signals=signals,
            network=network,
            evidence=evidence,
            llm_risk_score=llm_score,
        )
        score = float(policy["final_score"])
        typology = _derive_typology(signals, network)
        recommendation_bundle = {
            "recommended_action": policy["final_action"],
            "recommendation": policy["final_recommendation"],
        }
        risk_drivers = _risk_drivers(signals, network, score)

        case_summary = {
            "alert_trigger": "High Velocity Transfer" if (base_tx.get("transaction_velocity_10min") or 0) >= 3 else "Transaction Anomaly",
            "typology": typology,
            "transaction_amount": base_tx.get("amount"),
            "deviation": f"{int(score * 100)}% risk score",
            "recommendation": recommendation_bundle["recommendation"],
            "risk_drivers": risk_drivers,
        }
        device_identity = {
            "device_type": base_tx.get("device_type"),
            "device_fingerprint": base_tx.get("device_id"),
            "ip_address": base_tx.get("ip_address"),
            "location": _normalize_location(base_tx.get("country")),
            "country": base_tx.get("country"),
        }
        supporting_transactions = [
            {
                "time": base_tx.get("timestamp"),
                "destination_account": base_tx.get("beneficiary_id"),
                "amount": base_tx.get("amount"),
                "balance_after": None,
            }
        ]
        beneficiary_analysis = {
            "beneficiary_id": base_tx.get("beneficiary_id"),
            "fraud_link": network["fraud_link"],
            "fraud_link_status": network["fraud_link_status"],
            "suspected_network_risk": network["suspected_network_risk"],
            "connections": None,
            "network_risk_label": network["network_risk_label"],
        }
        decision["typology"] = typology
        decision["recommended_action"] = recommendation_bundle["recommended_action"]
        decision["recommendation"] = recommendation_bundle["recommendation"]
        decision["risk_band"] = policy["final_band"]
        decision["hard_stop_triggered"] = policy["hard_stop_triggered"]
        decision["hard_stop_reasons"] = policy["hard_stop_reasons"]
        decision["signal_contributions"] = policy["contributions"]
        decision["risk_drivers"] = risk_drivers
        decision["decision_rationale"] = (
            f"Risk score {score:.2f} with key drivers: "
            + (", ".join(risk_drivers) if risk_drivers else "transaction anomaly pattern under review")
            + f". Recommended action: {recommendation_bundle['recommended_action']}."
        )
        inv["typology"] = typology
        inv["recommendation"] = recommendation_bundle["recommendation"]
        inv["risk_score"] = score
        inv["risk_band"] = policy["final_band"]

        payload["alert_id"] = inv.get("transaction_id")
        payload["customer_id"] = inv.get("account_id")
        payload["account_id"] = inv.get("account_id")
        payload["timestamp"] = base_tx.get("timestamp")
        payload["risk_score"] = score
        payload["case_summary"] = case_summary
        payload["device_identity"] = device_identity
        payload["signals"] = signals
        payload["supporting_transactions"] = supporting_transactions
        payload["beneficiary_analysis"] = beneficiary_analysis
        payload["decision"] = decision
        payload["investigation"] = inv
        return payload

    def _run_investigation_for_alert(
        alert_id: str,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute live investigation workflow for a single alert id with explicit logs.
        """
        logger.info("[INVESTIGATE] alert_id=%s", alert_id)
        logger.info("[INVESTIGATE] loading source data")
        llm_info = get_llm_runtime_info()
        logger.warning(
            "[INVESTIGATE] llm_runtime provider=%s model=%s api_key_present=%s",
            llm_info.get("provider"),
            llm_info.get("model"),
            llm_info.get("has_api_key"),
        )
        try:
            pipeline = _get_pipeline_singleton()
            result: Dict[str, Any] = pipeline.run_investigation(
                alert_id,
                progress_callback=progress_callback,
            ).to_dict()
            result = _augment_investigation_payload(result)
            result = jsonable_encoder(result)
            completed_at = datetime.utcnow().isoformat() + "Z"
            decision = result.get("decision") or {}
            result["investigation_status"] = "investigated"
            result["investigation_completed_at"] = completed_at
            result["timestamp"] = completed_at
            result["final_risk_score"] = result.get("risk_score")
            result["final_recommendation"] = decision.get("recommendation")
            result["queue_status"] = _map_decision_to_queue_status(decision)
            result["llm_provider_used"] = result.get("llm_provider_used") or llm_info.get("provider") or "openai"
            result["model_used"] = result.get("model_used") or llm_info.get("model")
            store = _get_investigation_store()
            store[alert_id] = {
                "data": result,
                "investigation_completed_at": completed_at,
            }
            logger.info("[INVESTIGATE] completed alert_id=%s", alert_id)
            return {"ok": True, "data": result}
        except Exception as exc:
            logger.exception("[INVESTIGATE] failed alert_id=%s error=%s", alert_id, exc)
            if os.getenv("INVESTIGATION_FALLBACK_MODE", "false").lower() == "true":
                logger.warning("[INVESTIGATE] using fallback mode for alert_id=%s", alert_id)
                repo = _get_repository_singleton()
                tx = repo.get_transaction(alert_id)
                if tx is None:
                    return {
                        "ok": False,
                        "error": "Transaction not found",
                        "details": f"Transaction {alert_id} not found in repository.",
                    }
                fallback_payload = {
                    "status": "success",
                    "investigation": {
                        "investigation_id": f"INV-{alert_id}",
                        "transaction_id": alert_id,
                        "account_id": tx.account_id,
                        "status": "COMPLETED",
                        "base_transaction": tx.model_dump(mode="json"),
                        "evidence": [],
                        "risk_score": _risk_from_tx(tx),
                        "typology": "Fallback Typology",
                        "recommendation": "Fallback recommendation from backend.",
                        "confidence": _risk_from_tx(tx),
                    },
                    "evidence": [],
                    "decision": {
                        "risk_score": _risk_from_tx(tx),
                        "typology": "Fallback Typology",
                        "recommendation": "Fallback recommendation from backend.",
                        "confidence": _risk_from_tx(tx),
                        "decision_rationale": "Crew execution failed; fallback mode enabled.",
                    },
                }
                safe_fallback = jsonable_encoder(_augment_investigation_payload(fallback_payload))
                completed_at = datetime.utcnow().isoformat() + "Z"
                safe_fallback["investigation_status"] = "investigated"
                safe_fallback["investigation_completed_at"] = completed_at
                safe_fallback["timestamp"] = completed_at
                safe_fallback["final_risk_score"] = safe_fallback.get("risk_score")
                safe_fallback["final_recommendation"] = (safe_fallback.get("decision") or {}).get("recommendation")
                safe_fallback["queue_status"] = _map_decision_to_queue_status(safe_fallback.get("decision"))
                safe_fallback["llm_provider_used"] = "deterministic"
                safe_fallback["model_used"] = llm_info.get("model")
                _get_investigation_store()[alert_id] = {
                    "data": safe_fallback,
                    "investigation_completed_at": completed_at,
                }
                return {"ok": True, "data": safe_fallback}
            return {
                "ok": False,
                "error": "Investigation execution failed",
                "details": f"{type(exc).__name__}: {exc}",
            }

    return app


# FastAPI expects an `app` module-level variable when run via `uvicorn app:app`
app = create_app()


def get_openai_api_key() -> str | None:
    """
    Helper used by downstream services/agents to access the OpenAI API key.

    The key is intentionally not validated here so that the application can
    start even when the key is missing (e.g. during local, non-agent tests).
    """
    return os.getenv("OPENAI_API_KEY")

