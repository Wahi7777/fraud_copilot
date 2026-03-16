import os

from datetime import datetime
from functools import lru_cache
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from models import EnrichedTransactionRecord
from services.data_repository import DataRepository, DataRepositoryConfig
from services.llm_provider import get_llm_runtime_info
from services.typology_classifier import TYPOLOGY_DEFINITIONS


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
        return "Account Takeover"
    if high_velocity:
        return "Velocity Fraud"
    if "transfer_amount" in signals or "balance_drain" in signals:
        return "Transaction Anomaly"
    return "Unknown / Mixed Pattern"


def _typology_definition(typology: str | None) -> str:
    token = str(typology or "").strip()
    return TYPOLOGY_DEFINITIONS.get(
        token,
        "Suspicious transaction pattern detected that deviates from normal account behavior.",
    )


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


def _base_recommendation_from_score(score: float) -> str:
    if score < 0.35:
        return "Clear"
    if score < 0.70:
        return "Escalate"
    return "Decline"


def _normalize_recommendation_value(value: Any, *, risk_score: float) -> str:
    token = str(value or "").strip().lower()
    if token == "clear" or "clear" in token:
        return "Clear"
    if token == "decline" or "decline" in token:
        return "Decline"
    if token == "escalate" or "escalate" in token:
        return "Escalate"
    return _base_recommendation_from_score(risk_score)


def _recommended_actions_for_recommendation(recommendation: str) -> List[str]:
    rec = str(recommendation or "").strip().lower()
    if rec == "clear":
        return [
            "Allow the transaction",
            "Record investigation outcome",
            "Close the case as false positive",
        ]
    if rec == "decline":
        return [
            "Block the transaction",
            "Place a temporary hold on the sender account",
            "Investigate linked beneficiary accounts",
            "Escalate to financial crime investigation if suspicious activity continues",
        ]
    return [
        "Perform enhanced customer verification",
        "Review recent transaction history",
        "Investigate beneficiary account activity",
        "Escalate to senior fraud analyst if risk persists",
    ]


def _workflow_from_recommendation(recommendation: str) -> str:
    return "Closed" if recommendation == "Clear" else "In Review"


def _agent_votes_from_trace(agent_trace: List[Dict[str, Any]]) -> Dict[str, str]:
    votes: Dict[str, str] = {}
    for step in agent_trace:
        agent = str(step.get("agent") or "").strip()
        if not agent:
            continue
        votes[agent] = "review"
    return votes


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
    logger.info("[ALERTS] queue_source_records=%d", len(records))

    catalog: List[Dict[str, Any]] = []
    for tx in records:
        risk_score = _risk_from_tx(tx)
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        recommendation = _base_recommendation_from_score(risk_score)
        workflow_status = "New"

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
                "status": workflow_status,
                "workflow_status": workflow_status,
                "recommendation": recommendation,
            }
        )
    logger.info("[ALERTS] catalog_size=%d", len(catalog))
    return catalog


@lru_cache(maxsize=1)
def _get_metrics_summary() -> Dict[str, Any]:
    repo = _get_repository_singleton()
    df = repo.load_paysim()
    total = int(len(df))
    flagged = int(df["is_fraud"].sum()) if not df.empty else 0
    logger.info("[METRICS] paysim_rows=%d flagged=%d", total, flagged)

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

    # Optional static mount for assets under /frontend (JS, CSS, images, etc.).
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/health", tags=["system"])
    def health_check() -> dict:
        """
        Lightweight health endpoint used by tests and orchestrators.
        """
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def serve_root() -> Any:
        """
        Serve the main dashboard HTML for the root path.
        Falls back to a simple JSON message if the file is missing.
        """
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"status": "ok", "message": "Fraud Co-Pilot backend running."}

    @app.get("/app.js", include_in_schema=False)
    def serve_app_js() -> Any:
        """
        Serve the main frontend JavaScript bundle at /app.js for production.

        This ensures that existing script tags pointing to /app.js continue to
        work even when static assets are mounted under /frontend.
        """
        app_js = frontend_dir / "app.js"
        if app_js.exists():
            return FileResponse(app_js, media_type="application/javascript")
        raise HTTPException(status_code=404, detail="app.js not found")

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
        queue_status: Optional[str] = None  # legacy UI compatibility
        workflow_status: Optional[str] = None
        recommendation: Optional[str] = None
        decision_confidence: Optional[float] = None
        decision_reason: Optional[List[str]] = None
        recommended_actions: Optional[List[str]] = None
        fraud_typology: Optional[str] = None
        typology_confidence: Optional[float] = None
        typology_definition: Optional[str] = None
        typology_reason: Optional[List[str]] = None
        triggered_signals: Optional[List[str]] = None
        signal_breakdown: Optional[Dict[str, float]] = None
        agent_votes: Optional[Dict[str, str]] = None
        override_applied: Optional[bool] = None
        override_reason: Optional[str] = None

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
        status: str  # legacy
        recommendation: str = "Escalate"
        workflow_status: str = "New"
        investigation_status: str = "new"
        investigation_completed_at: Optional[datetime] = None
        investigation_summary: Optional[str] = None
        final_risk_score: Optional[float] = None
        final_recommendation: Optional[str] = None
        decision_confidence: Optional[float] = None
        decision_reason: Optional[List[str]] = None
        recommended_actions: Optional[List[str]] = None
        fraud_typology: Optional[str] = None
        typology_confidence: Optional[float] = None
        typology_definition: Optional[str] = None
        typology_reason: Optional[List[str]] = None
        triggered_signals: Optional[List[str]] = None
        signal_breakdown: Optional[Dict[str, float]] = None
        agent_votes: Optional[Dict[str, str]] = None
        override_applied: Optional[bool] = None
        override_reason: Optional[str] = None
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
            recommendation = _normalize_recommendation_value(row.get("recommendation"), risk_score=risk_score)
            workflow_status = row.get("workflow_status") or "New"
            investigation_status = "new"
            investigation_completed_at = None
            investigation_summary = None
            final_risk_score = None
            final_recommendation = None
            decision_confidence = None
            decision_reason = None
            recommended_actions = None
            fraud_typology = None
            typology_confidence = None
            typology_definition = None
            typology_reason = None
            triggered_signals = None
            signal_breakdown = None
            agent_votes = None
            override_applied = None
            override_reason = None
            has_cached_investigation = False
            last_investigation_id = None

            if store_entry:
                data = store_entry.get("data") or {}
                decision = data.get("decision") or {}
                risk_score = float(data.get("risk_score", risk_score))
                recommendation = _normalize_recommendation_value(
                    data.get("recommendation")
                    or decision.get("recommendation")
                    or data.get("final_recommendation")
                    or recommendation,
                    risk_score=risk_score,
                )
                workflow_status = data.get("workflow_status") or "In Review"
                investigation_status = "investigated"
                investigation_completed_at = store_entry.get("investigation_completed_at")
                investigation_summary = (
                    (data.get("case_summary") or {}).get("typology")
                    or decision.get("decision_rationale")
                )
                final_risk_score = risk_score
                final_recommendation = decision.get("recommendation")
                decision_confidence = data.get("decision_confidence") or decision.get("decision_confidence")
                decision_reason = data.get("decision_reason") or decision.get("decision_reason")
                if isinstance(decision_reason, str):
                    decision_reason = [decision_reason]
                recommended_actions = data.get("recommended_actions") or decision.get("recommended_actions")
                if isinstance(recommended_actions, str):
                    recommended_actions = [recommended_actions]
                if not isinstance(recommended_actions, list):
                    recommended_actions = None
                fraud_typology = data.get("fraud_typology") or decision.get("fraud_typology") or decision.get("typology")
                typology_confidence = data.get("typology_confidence") or decision.get("typology_confidence")
                typology_definition = data.get("typology_definition") or decision.get("typology_definition")
                typology_reason = data.get("typology_reason") or decision.get("typology_reason")
                if isinstance(typology_reason, str):
                    typology_reason = [typology_reason]
                if not isinstance(typology_reason, list):
                    typology_reason = None
                triggered_signals = data.get("triggered_signals") or decision.get("triggered_signals")
                if not isinstance(triggered_signals, list):
                    triggered_signals = None
                signal_breakdown = data.get("signal_breakdown") or decision.get("signal_breakdown")
                if not isinstance(signal_breakdown, dict):
                    signal_breakdown = None
                agent_votes = data.get("agent_votes") or decision.get("agent_votes")
                if not isinstance(agent_votes, dict):
                    agent_votes = None
                override_applied = data.get("override_applied") if data.get("override_applied") is not None else decision.get("override_applied")
                override_reason = data.get("override_reason") or decision.get("override_reason")
                has_cached_investigation = True
                last_investigation_id = (data.get("investigation") or {}).get("investigation_id")

            if min_risk_score is not None and risk_score < min_risk_score:
                continue
            if max_risk_score is not None and risk_score > max_risk_score:
                continue
            row_payload = dict(row)
            row_payload["risk_score"] = risk_score
            row_payload["risk_level"] = "HIGH" if risk_score >= 0.7 else "MEDIUM" if risk_score >= 0.4 else "LOW"
            row_payload["recommendation"] = recommendation
            row_payload["workflow_status"] = workflow_status
            row_payload["status"] = workflow_status
            row_payload["queue_status"] = recommendation
            if investigation_completed_at:
                row_payload["timestamp"] = investigation_completed_at
            row_payload["investigation_status"] = investigation_status
            row_payload["investigation_completed_at"] = investigation_completed_at
            row_payload["investigation_summary"] = investigation_summary
            row_payload["final_risk_score"] = final_risk_score
            row_payload["final_recommendation"] = final_recommendation
            row_payload["decision_confidence"] = decision_confidence
            row_payload["decision_reason"] = decision_reason
            row_payload["recommended_actions"] = recommended_actions
            row_payload["fraud_typology"] = fraud_typology
            row_payload["typology_confidence"] = typology_confidence
            row_payload["typology_definition"] = typology_definition
            row_payload["typology_reason"] = typology_reason
            row_payload["triggered_signals"] = triggered_signals
            row_payload["signal_breakdown"] = signal_breakdown
            row_payload["agent_votes"] = agent_votes
            row_payload["override_applied"] = override_applied
            row_payload["override_reason"] = override_reason
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
        score = _to_float(inv.get("risk_score", decision.get("risk_score")))
        typology = (
            decision.get("fraud_typology")
            or decision.get("typology")
            or inv.get("fraud_typology")
            or inv.get("typology")
            or _derive_typology(signals, network)
        )
        typology_confidence = _to_float(
            decision.get("typology_confidence", inv.get("typology_confidence")),
            default=0.6,
        )
        typology_definition = (
            decision.get("typology_definition")
            or inv.get("typology_definition")
            or _typology_definition(typology)
        )
        typology_reason = decision.get("typology_reason") or inv.get("typology_reason") or []
        if isinstance(typology_reason, str):
            typology_reason = [typology_reason]
        if not isinstance(typology_reason, list):
            typology_reason = []
        typology_reason = [str(x) for x in typology_reason if str(x).strip()][:4]
        recommendation = _normalize_recommendation_value(
            decision.get("recommendation") or inv.get("recommendation"),
            risk_score=score,
        )
        decision_confidence = _to_float(
            decision.get("decision_confidence", decision.get("confidence", inv.get("confidence"))),
            default=max(0.45, min(0.95, 0.35 + 0.55 * score)),
        )
        decision_reason = decision.get("decision_reason")
        if isinstance(decision_reason, str):
            decision_reason = [decision_reason]
        if not isinstance(decision_reason, list):
            decision_reason = []
        decision_reason = [str(x) for x in decision_reason if str(x).strip()]
        recommended_actions = decision.get("recommended_actions") or payload.get("recommended_actions") or inv.get("recommended_actions")
        if isinstance(recommended_actions, str):
            recommended_actions = [recommended_actions]
        if not isinstance(recommended_actions, list) or not recommended_actions:
            recommended_actions = _recommended_actions_for_recommendation(recommendation)
        recommended_actions = [str(x).strip() for x in recommended_actions if str(x).strip()]
        triggered_signals = payload.get("triggered_signals") or inv.get("triggered_signals") or decision.get("triggered_signals") or []
        if not isinstance(triggered_signals, list):
            triggered_signals = []
        triggered_signals = [str(s) for s in triggered_signals]
        signal_breakdown = payload.get("signal_breakdown") or inv.get("signal_breakdown") or {}
        if not isinstance(signal_breakdown, dict):
            signal_breakdown = {}
        agent_votes = decision.get("agent_votes")
        if not isinstance(agent_votes, dict):
            agent_votes = _agent_votes_from_trace(payload.get("agent_trace") or [])
        override_applied = bool(decision.get("override_applied", False))
        override_reason = decision.get("override_reason")
        workflow_status = _workflow_from_recommendation(recommendation)
        risk_drivers = _risk_drivers(signals, network, score)
        logger.warning(
            "[DECISION TRACE] alert_id=%s risk_score=%.3f signal_contributions=%s recommendation=%s confidence=%.3f triggered_signals=%s agent_votes=%s override_reason=%s",
            inv.get("transaction_id"),
            score,
            signal_breakdown,
            recommendation,
            decision_confidence,
            triggered_signals,
            agent_votes,
            override_reason,
        )

        case_summary = {
            "alert_trigger": "High Velocity Transfer" if (base_tx.get("transaction_velocity_10min") or 0) >= 3 else "Transaction Anomaly",
            "typology": typology,
            "typology_definition": typology_definition,
            "transaction_amount": base_tx.get("amount"),
            "deviation": f"{int(score * 100)}% risk score",
            "recommendation": recommendation,
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
        decision["fraud_typology"] = typology
        decision["typology_confidence"] = typology_confidence
        decision["typology_definition"] = typology_definition
        decision["typology_reason"] = typology_reason
        decision["recommendation"] = recommendation
        decision["decision_confidence"] = decision_confidence
        decision["decision_reason"] = decision_reason
        decision["recommended_actions"] = recommended_actions
        decision["triggered_signals"] = triggered_signals
        decision["signal_breakdown"] = signal_breakdown
        decision["agent_votes"] = agent_votes
        decision["override_applied"] = override_applied
        decision["override_reason"] = override_reason
        decision["risk_drivers"] = risk_drivers
        decision["decision_rationale"] = (
            f"Risk score {score:.2f} with key drivers: "
            + (", ".join(risk_drivers) if risk_drivers else "transaction anomaly pattern under review")
            + f". Recommendation: {recommendation}."
        )
        inv["typology"] = typology
        inv["fraud_typology"] = typology
        inv["typology_confidence"] = typology_confidence
        inv["typology_definition"] = typology_definition
        inv["typology_reason"] = typology_reason
        inv["recommendation"] = recommendation
        inv["recommended_actions"] = recommended_actions
        inv["risk_score"] = score
        inv["workflow_status"] = workflow_status
        inv["signal_breakdown"] = signal_breakdown
        inv["triggered_signals"] = triggered_signals

        payload["alert_id"] = inv.get("transaction_id")
        payload["customer_id"] = inv.get("account_id")
        payload["account_id"] = inv.get("account_id")
        payload["timestamp"] = base_tx.get("timestamp")
        payload["risk_score"] = score
        payload["fraud_typology"] = typology
        payload["typology_confidence"] = typology_confidence
        payload["typology_definition"] = typology_definition
        payload["typology_reason"] = typology_reason
        payload["workflow_status"] = workflow_status
        payload["recommendation"] = recommendation
        payload["decision_confidence"] = decision_confidence
        payload["decision_reason"] = decision_reason
        payload["recommended_actions"] = recommended_actions
        payload["triggered_signals"] = triggered_signals
        payload["signal_breakdown"] = signal_breakdown
        payload["agent_votes"] = agent_votes
        payload["override_applied"] = override_applied
        payload["override_reason"] = override_reason
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
            result["final_recommendation"] = _normalize_recommendation_value(
                result.get("recommendation") or decision.get("recommendation"),
                risk_score=float(result.get("risk_score") or 0.0),
            )
            result["workflow_status"] = result.get("workflow_status") or "In Review"
            result["queue_status"] = result["final_recommendation"]
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
                        "typology": "Unknown / Mixed Pattern",
                        "recommendation": "Fallback recommendation from backend.",
                        "confidence": _risk_from_tx(tx),
                    },
                    "evidence": [],
                    "decision": {
                        "risk_score": _risk_from_tx(tx),
                        "fraud_typology": "Unknown / Mixed Pattern",
                        "typology_confidence": 0.5,
                        "typology_definition": _typology_definition("Unknown / Mixed Pattern"),
                        "typology_reason": ["Fallback mode used due to investigation runtime failure."],
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
                safe_fallback["final_recommendation"] = _normalize_recommendation_value(
                    safe_fallback.get("recommendation") or (safe_fallback.get("decision") or {}).get("recommendation"),
                    risk_score=float(safe_fallback.get("risk_score") or 0.0),
                )
                safe_fallback["workflow_status"] = safe_fallback.get("workflow_status") or "In Review"
                safe_fallback["queue_status"] = safe_fallback["final_recommendation"]
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

