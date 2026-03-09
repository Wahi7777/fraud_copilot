from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import json
import logging
from datetime import datetime
from decimal import Decimal
import re

from crewai import Agent as CrewAgent, Crew, Process, Task as CrewTask
from fastapi.encoders import jsonable_encoder

from agents.behaviour_agent import BehaviourAgent
from agents.decision_agent import DecisionAgent
from agents.network_agent import NetworkAgent
from agents.pattern_agent import PatternAgent
from agents.typology_agent import TypologyAgent
from models import EvidenceItem, FraudSignalCode, InvestigationState
from services.evidence_registry import EvidenceRegistry
from services.llm_provider import get_llm

logger = logging.getLogger("fraud_api")

_VALID_SIGNAL_CODES = {code.value for code in FraudSignalCode}
_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def _contains_non_json_scalar(obj: Any) -> bool:
    """
    Return True if raw datetime/Decimal objects are still present.
    """
    if isinstance(obj, (datetime, Decimal)):
        return True
    if isinstance(obj, dict):
        return any(_contains_non_json_scalar(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_non_json_scalar(v) for v in obj)
    return False


def _preview_text(value: Any, max_len: int = 600) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...<truncated>"


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    """
    Best-effort extraction of JSON object from plain, fenced or prose-wrapped text.
    """
    if not text:
        return None

    # 1) Direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2) Parse fenced json blocks.
    for block in re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # 3) Parse first balanced object in prose.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _coerce_crew_output(raw: Any) -> Dict[str, Any]:
    """
    Normalize Crew output into a dict payload.
    Supports dicts, JSON strings, fenced JSON and CrewOutput-style objects.
    """
    logger.info("[INVESTIGATE] crew raw output type=%s", type(raw).__name__)
    logger.info("[INVESTIGATE] crew raw output preview=%s", _preview_text(raw))

    if isinstance(raw, dict):
        logger.info("[INVESTIGATE] crew parsed output preview=%s", _preview_text(raw))
        return raw

    candidates: List[Any] = [raw]
    for attr in ("raw", "result", "output", "final_output"):
        if hasattr(raw, attr):
            candidates.append(getattr(raw, attr))
    if hasattr(raw, "to_dict"):
        try:
            candidates.append(raw.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass

    for candidate in candidates:
        if isinstance(candidate, dict):
            logger.info("[INVESTIGATE] crew parsed output preview=%s", _preview_text(candidate))
            return candidate
        if isinstance(candidate, str):
            extracted = _extract_json_object(candidate)
            if extracted is not None:
                logger.info("[INVESTIGATE] crew parsed output preview=%s", _preview_text(extracted))
                return extracted

    raise ValueError("Crew output could not be normalized into a JSON object")


def _normalize_crew_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize key variants into the internal contract consumed by the pipeline.
    """
    pattern = raw.get("pattern_evidence")
    behaviour = raw.get("behaviour_evidence")
    network = raw.get("network_evidence")

    if not isinstance(pattern, list):
        pattern = raw.get("pattern") if isinstance(raw.get("pattern"), list) else []
    if not isinstance(behaviour, list):
        behaviour = raw.get("behavior_evidence")
        if not isinstance(behaviour, list):
            behaviour = raw.get("behaviour") if isinstance(raw.get("behaviour"), list) else []
    if not isinstance(network, list):
        network = raw.get("network") if isinstance(raw.get("network"), list) else []

    if not pattern and not behaviour and not network and isinstance(raw.get("evidence_items"), list):
        pattern = raw.get("evidence_items") or []

    decision = raw.get("decision")
    if not isinstance(decision, dict):
        decision = raw.get("final_decision") if isinstance(raw.get("final_decision"), dict) else None
    if not isinstance(decision, dict) and isinstance(raw.get("risk_score"), (int, float)):
        decision = {
            "risk_score": float(raw.get("risk_score", 0.0)),
            "typology": raw.get("typology"),
            "recommendation": raw.get("recommendation"),
            "confidence": float(raw.get("confidence", 0.0)),
            "decision_rationale": raw.get("decision_rationale") or raw.get("rationale"),
        }

    return {
        "pattern_evidence": pattern if isinstance(pattern, list) else [],
        "behaviour_evidence": behaviour if isinstance(behaviour, list) else [],
        "network_evidence": network if isinstance(network, list) else [],
        "typology": raw.get("typology"),
        "decision": decision,
    }


def _normalize_probability(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(score, 0.99))

@dataclass
class CrewRunResult:
    """
    Structured result returned by the InvestigationCrew.
    """

    state: InvestigationState
    evidence: List[EvidenceItem]
    decision: Dict[str, Any]
    agent_trace: List[Dict[str, str]]
    llm_provider_used: str
    model_used: str


class InvestigationCrew:
    """
    CrewAI-based orchestration layer for the fraud investigation pipeline.

    IMPORTANT:
    - CrewAI is used to model agents, tasks and the overall sequential process.
    - Actual reasoning and data processing are performed by the deterministic
      domain agents, in order to keep behaviour reproducible and auditable.
    - There is intentionally no dynamic delegation or tool use.
    """

    def __init__(
        self,
        pattern_agent: PatternAgent,
        behaviour_agent: BehaviourAgent,
        network_agent: NetworkAgent,
        typology_agent: TypologyAgent,
        decision_agent: DecisionAgent,
    ) -> None:
        self.pattern_agent_logic = pattern_agent
        self.behaviour_agent_logic = behaviour_agent
        self.network_agent_logic = network_agent
        self.typology_agent_logic = typology_agent
        self.decision_agent_logic = decision_agent

        llm = get_llm()

        # ------------------------------------------------------------------ #
        # CrewAI agents: bounded autonomy, explicit llm, no delegation.
        # ------------------------------------------------------------------ #
        self.pattern_agent = CrewAgent(
            role="Pattern Analyst",
            goal="Analyse transaction velocity and spikes and produce JSON evidence.",
            backstory="Specialist in identifying unusual velocity and spike patterns in payment flows.",
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.behaviour_agent = CrewAgent(
            role="Behaviour Analyst",
            goal="Analyse device, location and impossible-travel behaviour and produce JSON evidence.",
            backstory="Specialist in device intelligence and geo-anomaly detection.",
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.network_agent = CrewAgent(
            role="Network Analyst",
            goal="Inspect account graph relationships and mule-like patterns and produce JSON evidence.",
            backstory="Specialist in graph-based fraud pattern analysis.",
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.typology_agent = CrewAgent(
            role="Typology Classifier",
            goal="Map accumulated evidence to a single fraud typology label.",
            backstory="Specialist in fraud typology classification.",
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.decision_agent = CrewAgent(
            role="Decisioning Analyst",
            goal="Produce the final risk score, recommendation and rationale from evidence and typology.",
            backstory="Senior fraud decisioning analyst.",
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

        # ------------------------------------------------------------------ #
        # CrewAI tasks: descriptive stages in a fixed sequential workflow.
        # ------------------------------------------------------------------ #
        self.pattern_task = CrewTask(
            description="Perform pattern analysis on the transaction (velocity and spikes) and output JSON evidence.",
            agent=self.pattern_agent,
            expected_output="JSON with an 'evidence_items' list matching the EvidenceItem schema.",
        )
        self.behaviour_task = CrewTask(
            description="Perform behaviour analysis on device and geo signals and output JSON evidence.",
            agent=self.behaviour_agent,
            expected_output="JSON with an 'evidence_items' list matching the EvidenceItem schema.",
        )
        self.network_task = CrewTask(
            description="Perform network analysis on the account graph and output JSON evidence.",
            agent=self.network_agent,
            expected_output="JSON with an 'evidence_items' list matching the EvidenceItem schema.",
        )
        self.typology_task = CrewTask(
            description="Classify fraud typology based on accumulated evidence and output a typology string.",
            agent=self.typology_agent,
            expected_output="JSON with a 'typology' field.",
        )
        self.decision_task = CrewTask(
            description="Compute final risk score, recommendation and confidence and output a structured JSON decision.",
            agent=self.decision_agent,
            expected_output=(
                "JSON with keys: 'risk_score', 'typology', 'recommendation', 'confidence', 'decision_rationale'."
            ),
        )

        self.agents: Tuple[CrewAgent, ...] = (
            self.pattern_agent,
            self.behaviour_agent,
            self.network_agent,
            self.typology_agent,
            self.decision_agent,
        )
        self.tasks: Tuple[CrewTask, ...] = (
            self.pattern_task,
            self.behaviour_task,
            self.network_task,
            self.typology_task,
            self.decision_task,
        )
        self.process = Process.sequential

        self.crew = Crew(
            agents=list(self.agents),
            tasks=list(self.tasks),
            process=self.process,
        )

    def run(
        self,
        state: InvestigationState,
        registry: EvidenceRegistry,
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
    ) -> CrewRunResult:
        """
        Execute the investigation stages in a fixed sequential order using
        the CrewAI runtime. Deterministic services prepare a structured
        context which is then passed into the Crew, and all task outputs
        are expected to be structured JSON only.
        """
        tx = state.base_transaction
        model_used = getattr(get_llm(), "model_name", "gpt-4o-mini")
        llm_provider_used = "openai"
        agent_trace: List[Dict[str, str]] = []
        scoring_context = {
            "risk_score": state.risk_score,
            "signal_breakdown": state.signal_breakdown,
            "triggered_signals": state.triggered_signals,
            "synthetic_telemetry_signals": state.synthetic_telemetry_signals,
            "typology_candidate": state.typology,
            "typology_candidate_confidence": state.typology_confidence,
            "typology_candidate_definition": state.typology_definition,
            "typology_candidate_reason": state.typology_reason,
        }

        logger.warning("[AGENT START] pattern_agent")
        if progress_callback is not None:
            progress_callback({"agent": "pattern_agent", "status": "started"})
        pattern_payload = self._run_evidence_agent_openai(
            agent_name="pattern_agent",
            prompt=(
                "You are Pattern Agent for banking fraud. Analyze transaction anomalies using deterministic signal evidence. "
                "Return JSON only: {\"evidence_items\":[{source_agent,signal_code,severity,summary,details}]}."
            ),
            input_payload={
                "transaction": tx.model_dump(mode="json"),
                "account_id": state.account_id,
                "scoring_context": scoring_context,
                "existing_evidence": [e.model_dump(mode="json") for e in registry.list_all()],
            },
            fallback_fn=lambda: self.pattern_agent_logic.run(tx),
        )
        self._ingest_agent_evidence(
            agent_name="pattern_agent",
            payload=pattern_payload,
            registry=registry,
            trace=agent_trace,
            progress_callback=progress_callback,
        )

        logger.warning("[AGENT START] behaviour_agent")
        if progress_callback is not None:
            progress_callback({"agent": "behaviour_agent", "status": "started"})
        behaviour_payload = self._run_evidence_agent_openai(
            agent_name="behaviour_agent",
            prompt=(
                "You are Behaviour Agent for banking fraud. Analyze device/ip/location and synthetic telemetry context. "
                "Return JSON only: {\"evidence_items\":[{source_agent,signal_code,severity,summary,details}]}."
            ),
            input_payload={
                "transaction": tx.model_dump(mode="json"),
                "account_id": state.account_id,
                "scoring_context": scoring_context,
                "existing_evidence": [e.model_dump(mode="json") for e in registry.list_all()],
            },
            fallback_fn=lambda: self.behaviour_agent_logic.run(tx),
        )
        self._ingest_agent_evidence(
            agent_name="behaviour_agent",
            payload=behaviour_payload,
            registry=registry,
            trace=agent_trace,
            progress_callback=progress_callback,
        )

        network_context = self._network_context(tx)
        logger.warning("[AGENT START] network_agent")
        if progress_callback is not None:
            progress_callback({"agent": "network_agent", "status": "started"})
        network_payload = self._run_evidence_agent_openai(
            agent_name="network_agent",
            prompt=(
                "You are Network Agent for banking fraud. Analyze beneficiary/network graph context and graph-linked risk evidence. "
                "Return JSON only: {\"evidence_items\":[{source_agent,signal_code,severity,summary,details}]}."
            ),
            input_payload={
                "transaction": tx.model_dump(mode="json"),
                "network_context": network_context,
                "scoring_context": scoring_context,
                "existing_evidence": [e.model_dump(mode="json") for e in registry.list_all()],
            },
            fallback_fn=lambda: self.network_agent_logic.run(tx),
        )
        self._ingest_agent_evidence(
            agent_name="network_agent",
            payload=network_payload,
            registry=registry,
            trace=agent_trace,
            progress_callback=progress_callback,
        )

        evidence = registry.list_all()
        state.evidence = evidence

        logger.warning("[AGENT START] typology_agent")
        if progress_callback is not None:
            progress_callback({"agent": "typology_agent", "status": "started"})
        typology_payload = self._run_openai_json_call(
            agent_name="typology_agent",
            prompt=(
                "You are Typology Agent. Classify fraud typology from evidence and deterministic signal breakdown. "
                "Use only this controlled typology set: "
                "[\"Potential Mule Transfer\",\"Velocity Fraud\",\"Account Takeover\",\"Beneficiary Risk\","
                "\"Transaction Anomaly\",\"Structured Cash-Out Pattern\",\"Unknown / Mixed Pattern\"]. "
                "Return JSON only: {\"fraud_typology\":\"...\",\"typology_confidence\":0-1,"
                "\"typology_definition\":\"...\",\"typology_reason\":[\"...\"]}."
            ),
            input_payload={
                "evidence_items": [e.model_dump(mode="json") for e in evidence],
                "scoring_context": scoring_context,
            },
        )
        if not isinstance(typology_payload, dict) or not (
            typology_payload.get("fraud_typology") or typology_payload.get("typology")
        ):
            logger.warning("[OPENAI FALLBACK] agent=typology_agent reason=invalid_json_payload")
            llm_provider_used = "openai_fallback"
            typology_payload = self.typology_agent_logic.run(
                evidence,
                triggered_signals=state.triggered_signals,
                signal_breakdown=state.signal_breakdown,
                candidate_typology=state.typology,
            )
        logger.warning("[AGENT OUTPUT] typology_agent = %s", _preview_text(typology_payload))
        logger.warning("[AGENT END] typology_agent")
        typology = str(
            typology_payload.get("fraud_typology")
            or typology_payload.get("typology")
            or state.typology
            or "Unknown / Mixed Pattern"
        )
        typology_confidence = _normalize_probability(
            typology_payload.get("typology_confidence", state.typology_confidence or 0.6)
        )
        typology_definition = str(
            typology_payload.get("typology_definition")
            or state.typology_definition
            or "Suspicious transaction pattern detected that deviates from normal account behavior."
        )
        typology_reason = typology_payload.get("typology_reason")
        if isinstance(typology_reason, str):
            typology_reason = [typology_reason]
        if not isinstance(typology_reason, list):
            typology_reason = list(state.typology_reason or [])
        typology_reason = [str(r) for r in typology_reason][:4]
        state.typology = typology
        state.typology_confidence = typology_confidence
        state.typology_definition = typology_definition
        state.typology_reason = typology_reason
        if progress_callback is not None:
            progress_callback({"agent": "typology_agent", "status": "completed", "summary": typology})
        agent_trace.append({"agent": "typology_agent", "status": "completed", "summary": typology})

        logger.warning("[AGENT START] decision_agent")
        if progress_callback is not None:
            progress_callback({"agent": "decision_agent", "status": "started"})
        decision = self._run_openai_json_call(
            agent_name="decision_agent",
            prompt=(
                "You are Decision Agent. Given deterministic scoring evidence, typology, and agent evidence, produce final analyst decision JSON only. "
                "Recommendation MUST be one of [\"Clear\",\"Escalate\",\"Decline\"]. "
                "Return JSON schema: {\"risk_score\":0-1,\"recommendation\":\"Clear|Escalate|Decline\","
                "\"decision_confidence\":0-1,\"decision_reason\":[\"...\"],\"fraud_typology\":\"...\"}."
            ),
            input_payload={
                "typology": typology,
                "typology_confidence": typology_confidence,
                "typology_definition": typology_definition,
                "typology_reason": typology_reason,
                "evidence_items": [e.model_dump(mode="json") for e in evidence],
                "scoring_context": scoring_context,
            },
        )
        required = {"recommendation", "decision_confidence", "decision_reason", "fraud_typology"}
        if not isinstance(decision, dict) or not required.issubset(set(decision.keys())):
            logger.warning("[OPENAI FALLBACK] agent=decision_agent reason=invalid_json_payload")
            llm_provider_used = "openai_fallback"
            decision = self.decision_agent_logic.run(evidence, typology=typology)
        logger.warning("[AGENT OUTPUT] decision_agent = %s", _preview_text(decision))
        logger.warning("[AGENT END] decision_agent")

        normalized_risk = _normalize_probability(
            state.risk_score if state.risk_score is not None else decision.get("risk_score", 0.0)
        )
        decision["risk_score"] = normalized_risk
        decision["fraud_typology"] = decision.get("fraud_typology") or typology
        decision["typology_confidence"] = typology_confidence
        decision["typology_definition"] = typology_definition
        decision["typology_reason"] = typology_reason
        rec = str(decision.get("recommendation") or "Escalate").strip().lower()
        if rec.startswith("clear"):
            decision["recommendation"] = "Clear"
        elif rec.startswith("decline"):
            decision["recommendation"] = "Decline"
        else:
            decision["recommendation"] = "Escalate"
        decision["decision_confidence"] = _normalize_probability(
            decision.get("decision_confidence", decision.get("confidence", 0.0))
        )
        reasons = decision.get("decision_reason")
        if isinstance(reasons, str):
            reasons = [r.strip() for r in re.split(r"[;\n]", reasons) if r.strip()]
        if not isinstance(reasons, list):
            reasons = []
        decision["decision_reason"] = [str(r) for r in reasons][:5]
        state.risk_score = normalized_risk
        state.recommendation = decision.get("recommendation")
        state.confidence = decision["decision_confidence"]
        decision_summary = f"{decision.get('recommendation', 'n/a')} / risk_score={normalized_risk:.2f}"
        agent_trace.append({"agent": "decision_agent", "status": "completed", "summary": decision_summary})
        if progress_callback is not None:
            progress_callback({"agent": "decision_agent", "status": "completed", "summary": decision_summary})
            progress_callback({"status": "final", "recommendation": decision.get("recommendation"), "risk_score": normalized_risk})

        return CrewRunResult(
            state=state,
            evidence=evidence,
            decision=decision,
            agent_trace=agent_trace,
            llm_provider_used=llm_provider_used if llm_provider_used == "openai_fallback" else "openai",
            model_used=model_used,
        )

    def _run_evidence_agent_openai(
        self,
        *,
        agent_name: str,
        prompt: str,
        input_payload: Dict[str, Any],
        fallback_fn: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = self._run_openai_json_call(
            agent_name=agent_name,
            prompt=prompt,
            input_payload=input_payload,
        )
        if not isinstance(payload, dict) or "evidence_items" not in payload:
            logger.warning("[OPENAI FALLBACK] agent=%s reason=missing_evidence_items", agent_name)
            return fallback_fn()
        return payload

    def _run_openai_json_call(
        self,
        *,
        agent_name: str,
        prompt: str,
        input_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        llm = get_llm()
        model_name = getattr(llm, "model_name", "gpt-4o-mini")
        logger.warning("[OPENAI CALL START] agent=%s model=%s", agent_name, model_name)
        logger.warning("OPENAI MODEL INVOKED")
        safe_input = jsonable_encoder(input_payload)
        message = llm.invoke(
            [
                ("system", prompt + " Never output markdown, comments or prose."),
                ("user", json.dumps(safe_input)),
            ]
        )
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(content, dict):
            parsed = content
        else:
            parsed = _extract_json_object(str(content)) or {}
        logger.warning("[OPENAI CALL END] agent=%s model=%s", agent_name, model_name)
        logger.warning("[OPENAI CALL OK] agent=%s model=%s", agent_name, model_name)
        if not isinstance(parsed, dict):
            raise ValueError(f"OpenAI output for {agent_name} is not a JSON object")
        return parsed

    def _network_context(self, tx: Any) -> Dict[str, Any]:
        try:
            degree = self.network_agent_logic.graph_builder.get_beneficiary_degree(tx.beneficiary_id)
            path_to_flagged = self.network_agent_logic.graph_builder.has_path_to_flagged_node(tx.account_id)
            circular = self.network_agent_logic.graph_builder.detect_simple_circular_pattern(tx.account_id)
            return {
                "beneficiary_degree": degree,
                "path_to_flagged": path_to_flagged,
                "circular_pattern": circular,
            }
        except Exception:
            return {}

    def _ingest_agent_evidence(
        self,
        *,
        agent_name: str,
        payload: Dict[str, Any],
        registry: EvidenceRegistry,
        trace: List[Dict[str, str]],
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        logger.warning("[AGENT OUTPUT] %s = %s", agent_name, _preview_text(payload))
        evidence_items = payload.get("evidence_items") or []
        added = 0
        if isinstance(evidence_items, list):
            for raw in evidence_items:
                raw = self._sanitize_raw_evidence_item(raw, agent_name=agent_name)
                try:
                    item = EvidenceItem.model_validate(raw)
                except Exception:
                    continue
                before = len(registry.list_all())
                registry.append(item)
                after = len(registry.list_all())
                if after > before:
                    added += 1
        logger.warning("[AGENT END] %s", agent_name)
        trace.append(
            {
                "agent": agent_name,
                "status": "completed",
                "summary": f"{added} evidence items",
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "agent": agent_name,
                    "status": "completed",
                    "summary": f"{added} evidence items",
                }
            )

    def _sanitize_raw_evidence_item(self, raw: Any, *, agent_name: str) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        item = dict(raw)
        signal_code = str(item.get("signal_code") or "").upper()
        summary = str(item.get("summary") or "").lower()
        details = item.get("details")
        if not isinstance(details, dict):
            details = {"note": str(details)} if details is not None else None
        if signal_code not in _VALID_SIGNAL_CODES:
            signal_code = self._infer_signal_code(agent_name=agent_name, summary=summary)
        severity = str(item.get("severity") or "MEDIUM").upper()
        if severity not in _VALID_SEVERITIES:
            severity = "MEDIUM"
        return {
            "source_agent": str(item.get("source_agent") or agent_name),
            "signal_code": signal_code,
            "severity": severity,
            "summary": str(item.get("summary") or "Fraud risk indicator identified."),
            "details": details,
        }

    def _infer_signal_code(self, *, agent_name: str, summary: str) -> str:
        if "impossible" in summary or "travel" in summary:
            return "IMPOSSIBLE_TRAVEL"
        if "country" in summary or "geo" in summary or "location" in summary:
            return "NEW_COUNTRY"
        if "device" in summary or "browser" in summary or "fingerprint" in summary:
            return "NEW_DEVICE"
        if "link" in summary or "fraud account" in summary:
            return "LINKED_FRAUD_ACCOUNT"
        if "mule" in summary or "network" in summary:
            return "MULE_PATTERN"
        if agent_name == "network_agent":
            return "LINKED_FRAUD_ACCOUNT"
        if agent_name == "behaviour_agent":
            return "NEW_DEVICE"
        return "HIGH_VELOCITY"


