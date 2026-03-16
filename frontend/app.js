(() => {
  // In production (Vercel) we want same-origin API calls by default.
  // window.API_BASE can still override this for local development.
  const API_BASE = typeof window !== "undefined" && window.API_BASE ? window.API_BASE : "";
  const DEV_FALLBACK_MODAL = Boolean(window.DEV_FALLBACK_MODAL);
  const DEV_DEBUG_UI = Boolean(window.DEV_DEBUG_UI);

  const alertsTbody = document.getElementById("alerts-tbody");
  const alertsCountEl = document.getElementById("alerts-count");
  const alertsNewBadge = document.getElementById("alerts-new-badge");

  const kpiMap = {
    flagged: {
      value: document.getElementById("kpi-flagged-value"),
      trend: document.getElementById("kpi-flagged-trend"),
    },
    highRisk: {
      value: document.getElementById("kpi-highrisk-value"),
      trend: document.getElementById("kpi-highrisk-trend"),
    },
    open: {
      value: document.getElementById("kpi-open-value"),
      trend: document.getElementById("kpi-open-trend"),
    },
    escalated: {
      value: document.getElementById("kpi-escalated-value"),
      trend: document.getElementById("kpi-escalated-trend"),
    },
    fp: {
      value: document.getElementById("kpi-fp-value"),
      trend: document.getElementById("kpi-fp-trend"),
    },
    avgTime: {
      value: document.getElementById("kpi-avgtime-value"),
      trend: document.getElementById("kpi-avgtime-trend"),
    },
  };

  const modalEl = document.getElementById("investigation-modal");
  const modalTitle = document.getElementById("modal-title");
  const modalRiskPill = document.getElementById("modal-risk-pill");
  const modalMeta = document.getElementById("modal-meta");
  const modalAlertTrigger = document.getElementById("modal-alert-trigger");
  const modalTypologyPill = document.getElementById("modal-typology-pill");
  const modalTypologyTooltip = document.getElementById("modal-typology-tooltip");
  const modalAmount = document.getElementById("modal-amount");
  const modalDeviation = document.getElementById("modal-deviation");
  const modalRecommendation = document.getElementById("modal-recommendation");
  const modalAgentTrace = document.getElementById("modal-agent-trace");
  const modalLlmDebug = document.getElementById("modal-llm-debug");
  const modalDeviceType = document.getElementById("modal-device-type");
  const modalLocation = document.getElementById("modal-location");
  const modalDeviceFingerprint = document.getElementById("modal-device-fingerprint");
  const modalIpAddress = document.getElementById("modal-ip-address");
  const modalTypologySecondary = document.getElementById("modal-typology-secondary");
  const modalActionPrimary = document.getElementById("modal-action-primary");
  const modalFooterPrimary = document.getElementById("modal-footer-primary");
  const modalSignals = document.getElementById("modal-signals");
  const modalSupportingTbody = document.getElementById("modal-supporting-tbody");
  const modalNetworkRiskPill = document.getElementById("modal-network-risk-pill");
  const modalNetworkTbody = document.getElementById("modal-network-tbody");
  const modalNetworkGraph = document.getElementById("modal-network-graph");
  let recommendationEllipsisTimer = null;
  let recommendationEllipsisStep = 0;
  let tableDelegationBound = false;
  let expandedAlertId = null;
  const alertsById = new Map();
  const investigationsById = new Map();
  const alertOrder = [];
  const AGENT_SEQUENCE = [
    "pattern_agent",
    "behaviour_agent",
    "network_agent",
    "typology_agent",
    "decision_agent",
  ];
  const TYPOLOGY_DEFINITION_FALLBACK = {
    "Potential Mule Transfer":
      "Funds appear to be routed through intermediary beneficiary accounts in a pattern consistent with mule-network behavior.",
    "Velocity Fraud":
      "Rapid transaction activity in a short window suggests unusually high transfer velocity inconsistent with typical account behavior.",
    "Account Takeover":
      "Device, network, and location telemetry indicate possible unauthorized account access and control.",
    "Beneficiary Risk":
      "The destination beneficiary shows elevated risk based on historical or network-linked fraud indicators.",
    "Transaction Anomaly":
      "The transaction deviates from normal account behavior but lacks a stronger, specific fraud typology pattern.",
    "Structured Cash-Out Pattern":
      "Transfer and cash-out sequence suggests staged movement of funds for rapid extraction.",
    "Unknown / Mixed Pattern":
      "Suspicious transaction pattern detected that deviates from normal account behavior.",
  };

  function safe(fn, fallback = null) {
    try {
      return fn();
    } catch {
      return fallback;
    }
  }

  async function fetchJson(url, options = {}) {
    const resp = await fetch(url, options);
    if (!resp.ok) {
      const bodyText = await resp.text().catch(() => "");
      console.error("[UI] non-200 response", { url, status: resp.status, body: bodyText });
      throw new Error(`HTTP ${resp.status}: ${bodyText}`);
    }
    return resp.json();
  }

  async function fetchJsonWithStatus(url, options = {}) {
    const resp = await fetch(url, options);
    const bodyText = await resp.text().catch(() => "");
    let payload = {};
    try {
      payload = bodyText ? JSON.parse(bodyText) : {};
    } catch {
      payload = {};
    }
    if (!resp.ok) {
      console.error("[UI] non-200 response", { url, status: resp.status, body: payload });
      throw new Error(`HTTP ${resp.status}`);
    }
    return { status: resp.status, json: payload, text: bodyText };
  }

  function stopRecommendationLoadingUI() {
    if (recommendationEllipsisTimer) {
      clearInterval(recommendationEllipsisTimer);
      recommendationEllipsisTimer = null;
    }
    recommendationEllipsisStep = 0;
    if (modalRecommendation) modalRecommendation.classList.remove("is-investigating");
    console.log("[UI] recommendation loading state OFF");
  }

  function startRecommendationLoadingUI() {
    if (!modalRecommendation) return;
    stopRecommendationLoadingUI();
    modalRecommendation.classList.add("is-investigating");
    console.log("[UI] recommendation loading state ON");
    modalRecommendation.innerHTML = `
      <div class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-100 text-primary border border-blue-200 mb-2">
        AI Investigation Running
      </div>
      <div id="modal-recommendation-loading-text" class="text-blue-900 text-sm">Fraud AI Agent calculating...</div>
    `;
    recommendationEllipsisTimer = setInterval(() => {
      recommendationEllipsisStep = (recommendationEllipsisStep + 1) % 3;
      const dots = ".".repeat(recommendationEllipsisStep + 1);
      const node = document.getElementById("modal-recommendation-loading-text");
      if (node) node.textContent = `Fraud AI Agent calculating${dots}`;
    }, 450);
  }

  function setModalLoadingState(message = "Loading investigation...", isAgentInvestigation = false) {
    if (modalTitle) modalTitle.textContent = "Alert #...";
    if (modalRiskPill) modalRiskPill.textContent = "Loading";
    if (modalMeta) modalMeta.textContent = message;
    if (modalAlertTrigger) modalAlertTrigger.textContent = "-";
    if (modalTypologyPill) modalTypologyPill.textContent = "-";
    if (modalAmount) modalAmount.textContent = "$0.00";
    if (modalDeviation) modalDeviation.innerHTML = '<i class="fa-solid fa-arrow-trend-up"></i> -';
    if (isAgentInvestigation) startRecommendationLoadingUI();
    else {
      stopRecommendationLoadingUI();
      if (modalRecommendation) modalRecommendation.textContent = message;
    }
    if (modalDeviceType) modalDeviceType.textContent = "Not Available";
    if (modalLocation) modalLocation.textContent = "Not Available";
    if (modalDeviceFingerprint) modalDeviceFingerprint.textContent = "Not Available";
    if (modalIpAddress) modalIpAddress.textContent = "Not Available";
    if (modalSignals) {
      modalSignals.innerHTML = '<div class="bg-white p-4 rounded-lg border border-border shadow-sm text-sm text-secondaryText">Loading signals...</div>';
    }
    if (modalAgentTrace) {
      modalAgentTrace.classList.add("hidden");
      modalAgentTrace.innerHTML = "";
    }
    if (modalLlmDebug) {
      modalLlmDebug.classList.add("hidden");
      modalLlmDebug.textContent = "";
    }
    if (modalSupportingTbody) {
      modalSupportingTbody.innerHTML = '<tr><td colspan="4" class="px-4 py-3 text-secondaryText">Loading supporting transactions...</td></tr>';
    }
    if (modalNetworkTbody) {
      modalNetworkTbody.innerHTML = '<tr><td colspan="4" class="px-4 py-3 text-secondaryText">Loading beneficiary/network analysis...</td></tr>';
    }
    if (modalNetworkGraph) {
      modalNetworkGraph.innerHTML = '<div class="w-full h-full flex items-center justify-center text-xs text-secondaryText">Loading network graph...</div>';
    }
  }

  function setModalErrorState(message) {
    stopRecommendationLoadingUI();
    if (modalTitle) modalTitle.textContent = "Investigation unavailable";
    if (modalRiskPill) modalRiskPill.textContent = "Error";
    if (modalMeta) modalMeta.textContent = message;
    if (modalRecommendation) modalRecommendation.textContent = message;
    if (modalAgentTrace) {
      modalAgentTrace.classList.add("hidden");
      modalAgentTrace.innerHTML = "";
    }
    if (modalLlmDebug) {
      modalLlmDebug.classList.add("hidden");
      modalLlmDebug.textContent = "";
    }
    if (modalSignals) {
      modalSignals.innerHTML = '<div class="bg-white p-4 rounded-lg border border-border shadow-sm text-sm text-highRisk">Failed to load investigation signals.</div>';
    }
    if (modalSupportingTbody) {
      modalSupportingTbody.innerHTML = '<tr><td colspan="4" class="px-4 py-3 text-highRisk">No supporting transactions available.</td></tr>';
    }
    if (modalNetworkTbody) {
      modalNetworkTbody.innerHTML = '<tr><td colspan="4" class="px-4 py-3 text-highRisk">No beneficiary/network analysis available.</td></tr>';
    }
    if (modalNetworkGraph) {
      modalNetworkGraph.innerHTML = '<div class="w-full h-full flex items-center justify-center text-xs text-highRisk">Network graph unavailable.</div>';
    }
  }

  function formatCurrency(amount) {
    if (typeof amount !== "number" || Number.isNaN(amount)) return "$0.00";
    return `$${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function formatTimestamp(ts) {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return "-";
    return d.toLocaleString();
  }

  function formatTimestampStacked(ts) {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return { date: "Not Available", time: "" };
    const date = d.toLocaleDateString(undefined, {
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
    const time = d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    return { date, time };
  }

  function displayOrNA(value) {
    const text = String(value ?? "").trim();
    return text ? text : "Not Available";
  }

  function deriveReasonBullets(result, caseSummary = {}, decision = {}) {
    const fromDrivers = Array.isArray(caseSummary.risk_drivers) ? caseSummary.risk_drivers : [];
    if (fromDrivers.length) return fromDrivers.slice(0, 4);
    const listReasons = result?.decision_reason || decision?.decision_reason;
    if (Array.isArray(listReasons) && listReasons.length) {
      return listReasons.map((x) => String(x)).filter(Boolean).slice(0, 5);
    }

    const raw = String(result?.decision_reason || decision?.decision_reason || decision?.decision_rationale || "").trim();
    if (!raw) return ["Signals indicate elevated fraud risk; analyst review recommended."];

    const cleaned = raw
      .replace(/\{.*\}/g, "")
      .replace(/base=.*$/i, "")
      .replace(/votes=.*$/i, "")
      .replace(/override=.*$/i, "")
      .replace(/\s+/g, " ")
      .trim();
    const parts = cleaned
      .split(/[;|]/)
      .map((p) => p.trim())
      .filter((p) => p && !p.includes("=") && p.length > 8);
    return (parts.length ? parts : [cleaned]).slice(0, 4);
  }

  function deriveRecommendedActions(result, decision = {}, recommendation = "Escalate") {
    const fromBackend = result?.recommended_actions || decision?.recommended_actions;
    if (Array.isArray(fromBackend) && fromBackend.length) {
      return fromBackend.map((x) => String(x)).filter(Boolean);
    }
    const rec = normalizeRecommendation(recommendation);
    if (rec === "Clear") {
      return [
        "Allow the transaction",
        "Record investigation outcome",
        "Close the case as false positive",
      ];
    }
    if (rec === "Decline") {
      return [
        "Block the transaction",
        "Place a temporary hold on the sender account",
        "Investigate linked beneficiary accounts",
        "Escalate to financial crime investigation if suspicious activity continues",
      ];
    }
    return [
      "Perform enhanced customer verification",
      "Review recent transaction history",
      "Investigate beneficiary account activity",
      "Escalate to senior fraud analyst if risk persists",
    ];
  }

  function normalizeSignalSeverity(value) {
    const token = String(value || "").toLowerCase();
    if (token === "high" || token === "critical") return "high";
    if (token === "medium") return "medium";
    return "low";
  }

  function getRiskBand(score) {
    const s = Number(score || 0);
    if (s >= 0.9) return "CRITICAL";
    if (s >= 0.7) return "HIGH";
    if (s >= 0.4) return "MEDIUM";
    return "LOW";
  }

  function normalizeRecommendation(value) {
    const token = String(value || "").trim().toLowerCase();
    if (token === "clear") return "Clear";
    if (token === "decline") return "Decline";
    if (token === "escalate") return "Escalate";
    if (token.includes("decline")) return "Decline";
    if (token.includes("clear")) return "Clear";
    if (token.includes("escalate")) return "Escalate";
    return "Escalate";
  }

  function recommendationPlan(recommendation) {
    const normalized = normalizeRecommendation(recommendation);
    if (normalized === "Decline") {
      return {
        recommendation: normalized,
        primaryActionLabel: "Decline Transaction",
        primaryActionIcon: "fa-ban",
      };
    }
    if (normalized === "Clear") {
      return {
        recommendation: normalized,
        primaryActionLabel: "Close Case",
        primaryActionIcon: "fa-check",
      };
    }
    return {
      recommendation: "Escalate",
      primaryActionLabel: "Escalate Case",
      primaryActionIcon: "fa-arrow-up-right-dots",
    };
  }

  function applyPrimaryActionUI(recommendation, plan) {
    if (!modalActionPrimary || !modalFooterPrimary) return;
    const rec = normalizeRecommendation(recommendation);
    const bandClass =
      rec === "Decline"
        ? "bg-highRisk hover:bg-red-600"
        : rec === "Escalate"
        ? "bg-amber-500 hover:bg-amber-600"
        : "bg-green-600 hover:bg-green-700";
    modalActionPrimary.className = `text-white ${bandClass} border border-transparent px-4 py-2 rounded-lg text-sm font-medium shadow-sm transition-colors flex items-center`;
    modalActionPrimary.innerHTML = `<i class="fa-solid ${plan.primaryActionIcon} mr-2"></i>${plan.primaryActionLabel}`;

    modalFooterPrimary.className = `w-full inline-flex justify-center rounded-lg border border-transparent shadow-sm px-4 py-2 ${bandClass} text-base font-medium text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:ml-3 sm:w-auto sm:text-sm`;
    modalFooterPrimary.textContent = plan.primaryActionLabel;
  }

  function normalizeSignalsPayload(rawSignals, baseTx = {}, evidenceList = []) {
    console.log("[UI] raw signals payload", rawSignals);
    const normalized = {};

    if (rawSignals && !Array.isArray(rawSignals) && typeof rawSignals === "object") {
      Object.entries(rawSignals).forEach(([key, value]) => {
        if (!value || typeof value !== "object") return;
        normalized[key] = {
          label: String(value.label ?? ""),
          value: String(value.value ?? ""),
          detail: String(value.detail ?? ""),
          severity: normalizeSignalSeverity(value.severity),
        };
      });
    }

    if (Array.isArray(rawSignals)) {
      rawSignals.forEach((signal) => {
        const code = String(signal?.signal_code || "").toUpperCase();
        const detail = String(signal?.summary || "Detected by investigation");
        const severity = normalizeSignalSeverity(signal?.severity);
        if (code.includes("VELOCITY") || code.includes("SPIKE")) {
          normalized.velocity = normalized.velocity || {
            value: `${baseTx.transaction_velocity_10min ?? 0} tx / 10m`,
            detail,
            severity,
          };
        } else if (code.includes("BALANCE") || code.includes("DRAIN")) {
          normalized.balance_drain = normalized.balance_drain || {
            label: "Balance Drain",
            value: baseTx.amount ? formatCurrency(baseTx.amount) : "Balance anomaly",
            detail,
            severity,
          };
        } else if (code.includes("AMOUNT") || code.includes("TRANSFER")) {
          normalized.transfer_amount = normalized.transfer_amount || {
            label: "Transfer Amount",
            value: baseTx.amount ? formatCurrency(baseTx.amount) : "Transfer amount",
            detail,
            severity,
          };
        } else if (code.includes("DEVICE") || code.includes("FINGERPRINT")) {
          normalized.device_reuse = normalized.device_reuse || {
            label: "Device Risk",
            value: baseTx.device_risk_score ? `${Math.round(baseTx.device_risk_score * 100)}%` : "Device risk",
            detail,
            severity,
          };
        } else if (code.includes("GEO") || code.includes("TRAVEL") || code.includes("LOCATION")) {
          normalized.geo_anomaly = normalized.geo_anomaly || {
            label: "Geo Anomaly",
            value: baseTx.geo_distance_jump_km ? `${Math.round(baseTx.geo_distance_jump_km)} km` : "Geo anomaly",
            detail,
            severity,
          };
        }
      });
    }

    if (!Object.keys(normalized).length) {
      const velocity = Number(baseTx.transaction_velocity_10min || 0);
      if (velocity > 0) {
        normalized.velocity = {
          label: "Velocity",
          value: `${velocity.toFixed(0)} tx / 10m`,
          detail: `${Number(baseTx.transaction_velocity_1hr || 0).toFixed(0)} tx in last hour`,
          severity: velocity >= 3 ? "high" : velocity >= 1 ? "medium" : "low",
        };
      }
      const deviceRisk = Number(baseTx.device_risk_score || 0);
      if (deviceRisk > 0) {
        normalized.device_reuse = {
          label: "Device Risk",
          value: `${Math.round(deviceRisk * 100)}%`,
          detail: "Device risk score from enrichment",
          severity: deviceRisk >= 0.75 ? "high" : deviceRisk >= 0.5 ? "medium" : "low",
        };
      }
      const geoJump = Number(baseTx.geo_distance_jump_km || 0);
      if (geoJump > 0 || baseTx.impossible_travel_flag) {
        normalized.geo_anomaly = {
          label: "Geo Anomaly",
          value: baseTx.impossible_travel_flag ? "Impossible travel" : `${Math.round(geoJump)} km`,
          detail: "Location anomaly from recent activity",
          severity: baseTx.impossible_travel_flag || geoJump >= 1500 ? "high" : "medium",
        };
      }
      const amount = Number(baseTx.amount || 0);
      if (amount > 0) {
        normalized.transfer_amount = {
          label: "Transfer Amount",
          value: formatCurrency(amount),
          detail: "Outbound transfer amount under review",
          severity: amount >= 5000 ? "high" : amount >= 1000 ? "medium" : "low",
        };
      }
    }

    if (Array.isArray(evidenceList) && evidenceList.length) {
      evidenceList.forEach((ev) => {
        const detail = String(ev?.summary || "").trim();
        if (!detail) return;
        const code = String(ev?.signal_code || "").toUpperCase();
        if (code.includes("VELOCITY") && normalized.velocity && !normalized.velocity.detail) normalized.velocity.detail = detail;
        if (code.includes("DEVICE") && normalized.device_reuse && !normalized.device_reuse.detail) normalized.device_reuse.detail = detail;
        if (code.includes("GEO") && normalized.geo_anomaly && !normalized.geo_anomaly.detail) normalized.geo_anomaly.detail = detail;
      });
    }

    console.log("[UI] normalized signals payload", normalized);
    return normalized;
  }

  function renderSignalCards(signalMap) {
    if (!modalSignals) return;
    const entries = Object.entries(signalMap || {}).filter(([, v]) => v && v.value);
    if (!entries.length) {
      modalSignals.innerHTML = '<div class="bg-white p-4 rounded-lg border border-border shadow-sm text-sm text-secondaryText">No investigation signals returned.</div>';
      return;
    }

    const iconByKey = {
      velocity: "fa-gauge-high",
      balance_drain: "fa-wallet",
      transfer_amount: "fa-money-bill-transfer",
      device_reuse: "fa-mobile-screen",
      geo_anomaly: "fa-globe",
    };
    const labelByKey = {
      velocity: "Velocity",
      balance_drain: "Balance Drain",
      transfer_amount: "Transfer Amount",
      device_reuse: "Device Reuse",
      geo_anomaly: "Geo Anomaly",
    };
    modalSignals.innerHTML = "";
    const currentRiskBand = modalRiskPill?.textContent?.toLowerCase().includes("high")
      ? "HIGH"
      : modalRiskPill?.textContent?.toLowerCase().includes("medium")
      ? "MEDIUM"
      : "LOW";
    const minSeverityRank = currentRiskBand === "LOW" ? 0 : 1;
    entries.forEach(([key, sig]) => {
      const rawSeverity = normalizeSignalSeverity(sig.severity);
      const severityRank = { low: 0, medium: 1, high: 2 }[rawSeverity] ?? 0;
      const severity = ["low", "medium", "high"][Math.max(severityRank, minSeverityRank)];
      const borderClass = severity === "high" ? "border-l-highRisk" : severity === "medium" ? "border-l-mediumRisk" : "border-l-lowRisk";
      const textClass = severity === "high" ? "text-highRisk" : severity === "medium" ? "text-mediumRisk" : "text-lowRisk";
      const div = document.createElement("div");
      div.className = `bg-white p-4 rounded-lg border border-l-4 ${borderClass} border-border shadow-sm`;
      div.innerHTML = `
        <div class="flex items-center gap-2 mb-2">
          <i class="fa-solid ${iconByKey[key] || "fa-circle-exclamation"} ${textClass}"></i>
          <span class="text-sm font-semibold text-text">${sig.label || labelByKey[key] || key.replaceAll("_", " ")}</span>
        </div>
        <div class="text-2xl font-bold text-text mb-1">${sig.value}</div>
        <p class="text-xs text-secondaryText">${sig.detail || "Signal detected"}</p>
      `;
      modalSignals.appendChild(div);
    });
  }

  function renderBeneficiaryGraph(beneficiary = {}, sourceId = "You") {
    if (!modalNetworkGraph) return;
    const benId = String(beneficiary.beneficiary_id || "Beneficiary");
    const benShort = benId.length > 12 ? `${benId.slice(0, 12)}...` : benId;
    const riskLabel = String(beneficiary.network_risk_label || "Low Network Risk");
    const riskTone = beneficiary.fraud_link
      ? "text-highRisk"
      : beneficiary.suspected_network_risk
      ? "text-amber-700"
      : "text-green-700";
    const beneficiaryNodeTone = beneficiary.fraud_link
      ? "bg-red-100 border-highRisk text-highRisk"
      : beneficiary.suspected_network_risk
      ? "bg-amber-100 border-amber-500 text-amber-700"
      : "bg-green-100 border-green-500 text-green-700";

    const connectionCount = Math.max(0, Number(beneficiary.connections || 0));
    const extraNodes = Array.from({ length: Math.min(connectionCount, 3) }).map((_, i) => ({
      top: 18 + i * 30,
      label: `C${i + 1}`,
    }));

    modalNetworkGraph.innerHTML = `
      <div class="absolute inset-0 opacity-10" style="background-image: radial-gradient(#2563EB 1px, transparent 1px); background-size: 10px 10px;"></div>
      <div class="relative w-full h-full">
        <div class="absolute left-[14%] top-1/2 -translate-y-1/2 flex flex-col items-center">
          <div class="w-11 h-11 rounded-full bg-blue-100 border-2 border-primary flex items-center justify-center text-primary text-xs font-bold shadow-sm">You</div>
          <div class="mt-2 text-[11px] text-secondaryText max-w-[90px] text-center break-words">${sourceId}</div>
        </div>
        <div class="absolute left-[26%] right-[35%] top-1/2 -translate-y-1/2 h-[2px] bg-red-300"></div>
        <div class="absolute right-[18%] top-1/2 -translate-y-1/2 flex flex-col items-center">
          <div class="w-12 h-12 rounded-full ${beneficiaryNodeTone} border-2 flex items-center justify-center text-xs font-bold shadow">${benShort}</div>
          <div class="mt-2 text-[11px] font-semibold ${riskTone} max-w-[130px] text-center leading-4 break-words">${riskLabel}</div>
        </div>
        ${extraNodes
          .map(
            (n) => `
          <div class="absolute right-[5%]" style="top:${n.top}%;">
            <div class="w-7 h-7 rounded-full bg-gray-100 border border-border flex items-center justify-center text-[10px] text-secondaryText font-semibold">${n.label}</div>
          </div>
          <div class="absolute right-[17%] h-[1px] bg-gray-300" style="top:${n.top + 6}%; width:9%;"></div>
        `
          )
          .join("")}
      </div>
    `;
  }

  function deriveRiskLevel(score) {
    if (score >= 0.7) return "HIGH";
    if (score >= 0.4) return "MEDIUM";
    return "LOW";
  }

  function mapRecommendationToQueueStatus(recommendation) {
    return normalizeRecommendation(recommendation);
  }

  function mapRiskLevelToBadgeClass(riskLevel) {
    const value = String(riskLevel || "").toLowerCase();
    if (value.includes("high") || value.includes("confirmed")) return "bg-red-100 text-highRisk border-red-200";
    if (value.includes("medium") || value.includes("suspected")) return "bg-amber-100 text-amber-700 border-amber-200";
    if (value.includes("low")) return "bg-green-100 text-green-700 border-green-200";
    return "bg-gray-100 text-secondaryText border-gray-200";
  }

  function typologyDefinitionFor(result, decision, inv, typology) {
    return (
      result.typology_definition ||
      decision.typology_definition ||
      inv.typology_definition ||
      TYPOLOGY_DEFINITION_FALLBACK[String(typology || "").trim()] ||
      TYPOLOGY_DEFINITION_FALLBACK["Unknown / Mixed Pattern"]
    );
  }

  function statusBadgeClasses(statusText) {
    const value = String(statusText || "").toLowerCase();
    if (value.includes("decline")) {
      return "bg-red-100 text-highRisk border-red-200";
    }
    if (value.includes("escalate") || value.includes("review")) return "bg-amber-100 text-amber-700 border-amber-200";
    if (value.includes("clear") || value.includes("closed") || value.includes("new")) {
      return "bg-green-100 text-green-700 border-green-200";
    }
    return "bg-gray-100 text-secondaryText border-gray-200";
  }

  function investigateButtonLabel(status) {
    if (status === "investigating") return "Investigating...";
    if (status === "investigated") return "View Investigation";
    if (status === "failed") return "Retry Investigation";
    return "Investigate";
  }

  function normalizeAlert(item) {
    const normalizedScore = Number(item.final_risk_score ?? item.risk_score ?? 0);
    const mappedRecommendation = mapRecommendationToQueueStatus(
      item.recommendation ||
        item.final_recommendation ||
        item?.decision?.recommendation ||
        item.queue_status
    );
    const mappedWorkflowStatus =
      item.workflow_status ||
      (item.investigation_status === "investigated" ? "In Review" : "New");
    return {
      ...item,
      investigation_status: item.investigation_status || "new",
      investigation_completed_at: item.investigation_completed_at || null,
      investigation_summary: item.investigation_summary || null,
      final_risk_score: item.final_risk_score ?? null,
      final_recommendation: item.final_recommendation || null,
      has_cached_investigation: Boolean(item.has_cached_investigation),
      last_investigation_id: item.last_investigation_id || null,
      recommendation: mappedRecommendation,
      workflow_status: mappedWorkflowStatus,
      queue_status: mappedRecommendation,
    };
  }

  function completenessScore(alert) {
    let score = 0;
    if (alert.investigation_status === "investigated") score += 5;
    if (alert.has_cached_investigation) score += 3;
    if (alert.final_risk_score != null) score += 2;
    if (alert.final_recommendation) score += 2;
    if (alert.investigation_completed_at) score += 1;
    return score;
  }

  function dedupeAlertsForQueue(items) {
    const incoming = Array.isArray(items) ? items : [];
    console.log("[UI] alerts returned by API:", incoming.length);

    // Stage 1: exact dedupe by alert_id (keep latest/most complete payload).
    const byId = new Map();
    const exactRemoved = [];
    incoming.forEach((raw, idx) => {
      const id = raw?.alert_id || raw?.transaction_id;
      if (!id) return;
      const normalized = normalizeAlert(raw);
      const existing = byId.get(id);
      if (!existing) {
        byId.set(id, { ...normalized, __idx: idx });
        return;
      }
      const takeIncoming =
        completenessScore(normalized) > completenessScore(existing) ||
        idx > (existing.__idx ?? -1);
      if (takeIncoming) {
        byId.set(id, { ...normalized, __idx: idx });
        exactRemoved.push(id);
      } else {
        exactRemoved.push(id);
      }
    });
    const exactList = Array.from(byId.values()).map(({ __idx, ...rest }) => rest);
    console.log("[UI] alerts after exact dedupe:", exactList.length);
    console.log("[UI] exact dedupe removed/merged alert_ids:", [...new Set(exactRemoved)]);

    // Stage 2: near-duplicate suppression for display credibility.
    // Intentionally ignores customer/account so repeated-looking alerts across
    // different customers collapse in MVP queue presentation.
    const nearMap = new Map();
    const nearRemoved = [];
    const suppressedGroupKeys = new Set();
    exactList.forEach((item, idx) => {
      const ts = item.timestamp ? new Date(item.timestamp) : null;
      const timestampBucket =
        ts && !Number.isNaN(ts.getTime())
          ? `${ts.getUTCFullYear()}-${String(ts.getUTCMonth() + 1).padStart(2, "0")}-${String(ts.getUTCDate()).padStart(2, "0")}T${String(ts.getUTCHours()).padStart(2, "0")}:${String(ts.getUTCMinutes()).padStart(2, "0")}`
          : "-";
      const amountValue = Number(item.amount || 0);
      const amountBucket = String(Math.round(amountValue / 100) * 100);
      const riskBucket = item.risk_score != null ? String(Math.round(Number(item.risk_score) * 10) / 10) : "-";
      const nearKey = [
        amountBucket,
        timestampBucket,
        item.alert_type || "-",
        riskBucket,
      ].join("|");

      const existing = nearMap.get(nearKey);
      if (!existing) {
        nearMap.set(nearKey, { ...item, __idx: idx });
        return;
      }
      const existingInvestigated = existing.investigation_status === "investigated";
      const incomingInvestigated = item.investigation_status === "investigated";
      let takeIncoming = false;
      if (incomingInvestigated !== existingInvestigated) {
        takeIncoming = incomingInvestigated;
      } else {
        const existingRisk = Number(existing.final_risk_score ?? existing.risk_score ?? 0);
        const incomingRisk = Number(item.final_risk_score ?? item.risk_score ?? 0);
        if (incomingRisk !== existingRisk) {
          takeIncoming = incomingRisk > existingRisk;
        } else {
          // Keep first encountered row when other criteria tie.
          takeIncoming = false;
        }
      }
      if (takeIncoming) {
        nearRemoved.push(existing.alert_id || existing.transaction_id || "unknown");
        suppressedGroupKeys.add(nearKey);
        nearMap.set(nearKey, { ...item, __idx: idx });
      } else {
        nearRemoved.push(item.alert_id || item.transaction_id || "unknown");
        suppressedGroupKeys.add(nearKey);
      }
    });
    const finalList = Array.from(nearMap.values()).map(({ __idx, ...rest }) => rest);
    console.log("[UI] alerts after near-duplicate suppression:", finalList.length);
    console.log("[UI] near-duplicate removed/merged alert_ids:", [...new Set(nearRemoved)]);
    console.log("[UI] near-duplicate grouping keys suppressed:", [...suppressedGroupKeys]);
    return finalList;
  }

  function renderAlerts() {
    if (!alertsTbody) return;
    alertsTbody.innerHTML = "";
    alertOrder.forEach((alertId, idx) => {
      const item = alertsById.get(alertId);
      if (!item) return;
      const tr = document.createElement("tr");
      const score = Number(item.final_risk_score ?? item.risk_score ?? 0);
      const riskLevel = item.risk_level || deriveRiskLevel(score);
      let borderClass = "border-l-4 border-l-lowRisk";
      if (riskLevel === "HIGH") borderClass = "border-l-4 border-l-highRisk";
      else if (riskLevel === "MEDIUM") borderClass = "border-l-4 border-l-mediumRisk";
      if (item.investigation_status === "investigated") borderClass += " ring-1 ring-green-200";

      tr.className = `alert-row expand-row bg-white hover:bg-gray-50 cursor-pointer group transition-colors ${borderClass}`;
      tr.dataset.alertId = item.alert_id || item.transaction_id || `row-${idx}`;
      tr.dataset.investigationStatus = item.investigation_status;
      const btnLabel = investigateButtonLabel(item.investigation_status);
      const recommendationLabel = normalizeRecommendation(item.recommendation || item.queue_status);
      const workflowLabel = item.workflow_status || "New";
      const badgeClasses = statusBadgeClasses(recommendationLabel);
      tr.innerHTML = `
        <td class="px-6 py-4 font-medium text-text">
          <div class="flex items-center gap-2">
            <i class="expand-btn expand-arrow fa-solid fa-chevron-right text-xs text-secondaryText transition-transform group-hover:text-primary"></i>
            ${item.alert_id || item.transaction_id}
          </div>
        </td>
        <td class="px-6 py-4">
          <div class="flex items-center gap-2">
            <div class="h-6 w-6 rounded-full bg-gray-200 flex items-center justify-center text-[10px] font-bold text-secondaryText">
              ${(item.account_id || "?").slice(-2)}
            </div>
            <span class="text-text font-medium">${item.account_id || "-"}</span>
          </div>
        </td>
        <td class="px-6 py-4 text-text">${item.alert_type || "-"}</td>
        <td class="px-6 py-4 font-medium text-text">${formatCurrency(item.amount)}</td>
        <td class="px-6 py-4">
          <div class="w-full bg-gray-200 rounded-full h-1.5 w-16 mb-1">
            <div class="${riskLevel === "HIGH" ? "bg-highRisk" : riskLevel === "MEDIUM" ? "bg-mediumRisk" : "bg-lowRisk"} h-1.5 rounded-full" style="width: ${Math.round(score * 100)}%"></div>
          </div>
          <span class="text-xs font-semibold ${
            riskLevel === "HIGH"
              ? "text-highRisk"
              : riskLevel === "MEDIUM"
              ? "text-mediumRisk"
              : "text-lowRisk"
          }">${score.toFixed(2)}</span>
        </td>
        <td class="px-6 py-4">${formatTimestamp(item.timestamp)}</td>
        <td class="px-6 py-4">
          <span class="${badgeClasses} text-xs font-medium px-2.5 py-0.5 rounded border">${recommendationLabel}</span>
          <div class="text-[11px] text-secondaryText mt-1">${workflowLabel}</div>
        </td>
        <td class="px-6 py-4 text-right">
          <button ${item.investigation_status === "investigating" ? "disabled" : ""} class="investigate-btn pointer-events-auto relative z-10 text-primary hover:text-blue-700 font-medium text-xs border border-blue-200 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded transition-colors ${item.investigation_status === "investigating" ? "opacity-60 cursor-not-allowed" : ""}">
            ${btnLabel}
          </button>
        </td>
      `;

      const previewRow = document.createElement("tr");
      previewRow.className = `expand-preview ${expandedAlertId === alertId ? "" : "hidden"} bg-gray-50/50`;
      previewRow.dataset.parentAlertId = tr.dataset.alertId;
      previewRow.innerHTML = `
        <td colspan="8" class="px-6 py-4">
          <div class="bg-white border border-border rounded-lg shadow-sm p-4 flex items-center justify-between">
            <div>
              <div class="text-sm font-semibold text-text">Alert Preview</div>
              <div class="text-xs text-secondaryText mt-0.5">
                Account ${item.account_id || "-"} transferred ${formatCurrency(item.amount)} to ${item.beneficiary_id || "-"}.
              </div>
              <div class="text-xs text-secondaryText mt-1">
                Investigation: ${item.investigation_status}${item.investigation_completed_at ? ` • Completed ${formatTimestamp(item.investigation_completed_at)}` : ""}
              </div>
            </div>
            <button class="preview-investigate investigate-btn text-xs font-medium text-primary hover:underline">${btnLabel}</button>
          </div>
        </td>
      `;

      alertsTbody.appendChild(tr);
      alertsTbody.appendChild(previewRow);
    });

    if (alertsCountEl) alertsCountEl.textContent = String(alertOrder.length);
    if (alertsNewBadge) alertsNewBadge.textContent = `${alertOrder.length} New`;
    if (expandedAlertId) decorateExpandedRow(expandedAlertId);
  }

  function collapseAllRows() {
    expandedAlertId = null;
    renderAlerts();
  }

  function toggleAlertRowById(alertId) {
    if (!alertId) return;
    expandedAlertId = expandedAlertId === alertId ? null : alertId;
    renderAlerts();
  }

  function decorateExpandedRow(alertId) {
    if (!alertsTbody || !alertId) return;
    const row = alertsTbody.querySelector(`.alert-row[data-alert-id="${alertId}"]`);
    if (!(row instanceof HTMLElement)) return;
    row.classList.add("expanded");
    const arrow = row.querySelector(".expand-arrow");
    if (arrow) arrow.classList.add("rotate-90");
  }

  function bindTableDelegation() {
    if (!alertsTbody || tableDelegationBound) return;
    tableDelegationBound = true;

    alertsTbody.addEventListener("click", (e) => {
      const target = e.target;
      if (!(target instanceof HTMLElement)) return;

      const investigateBtn = target.closest(".investigate-btn");
      if (investigateBtn instanceof HTMLButtonElement) {
        e.stopPropagation();
        const row = investigateBtn.closest("tr");
        if (!(row instanceof HTMLElement)) return;
        const parentRow = row.classList.contains("alert-row")
          ? row
          : row.previousElementSibling instanceof HTMLElement
          ? row.previousElementSibling
          : null;
        const alertId = parentRow?.dataset.alertId;
        if (!alertId) return;
        const alertItem = alertsById.get(alertId);
        if (!alertItem) return;

        if (alertItem.investigation_status === "investigated") {
          openCachedOrStoredInvestigation(alertId);
          return;
        }
        runInvestigationForAlert(alertId);
        return;
      }

      const expandBtn = target.closest(".expand-btn");
      const rowTarget = target.closest(".alert-row");
      const row = (expandBtn?.closest(".alert-row") || rowTarget);
      if (!(row instanceof HTMLElement)) return;
      const alertId = row.dataset.alertId;
      if (!alertId) return;

      const alertItem = alertsById.get(alertId);
      if (alertItem?.investigation_status === "investigated") {
        openCachedOrStoredInvestigation(alertId);
        return;
      }
      toggleAlertRowById(alertId);
    });
  }

  async function loadAlerts() {
    try {
      const data = await fetchJson(`${API_BASE}/alerts?limit=25`);
      const cleanItems = dedupeAlertsForQueue(data.items || []);
      alertsById.clear();
      alertOrder.length = 0;
      cleanItems.forEach((item) => {
        const normalized = normalizeAlert(item);
        const id = normalized.alert_id || normalized.transaction_id;
        if (!id) return;
        alertsById.set(id, normalized);
        alertOrder.push(id);
      });
      renderAlerts();
    } catch (e) {
      console.error("Failed to load alerts", e);
    }
  }

  async function loadMetrics() {
    try {
      const m = await fetchJson(`${API_BASE}/metrics`);
      if (kpiMap.flagged.value) kpiMap.flagged.value.textContent = String(m.flagged_transactions ?? 0);
      if (kpiMap.highRisk.value) kpiMap.highRisk.value.textContent = String(m.high_risk_alerts ?? 0);
      if (kpiMap.open.value) kpiMap.open.value.textContent = String(m.open_investigations ?? 0);
      if (kpiMap.escalated.value) kpiMap.escalated.value.textContent = String(m.escalated_cases ?? 0);
      if (kpiMap.fp.value) kpiMap.fp.value.textContent = `${Math.round((m.false_positive_rate ?? 0) * 100)}%`;
      if (kpiMap.avgTime.value) kpiMap.avgTime.value.textContent = `${m.avg_resolution_time ?? 0}m`;
    } catch (e) {
      console.error("Failed to load metrics", e);
    }
  }

  function openModalLoading() {
    modalEl.classList.remove("hidden");
    document.body.classList.add("modal-open");
    setModalLoadingState();
  }

  function openModalLoadingForInvestigation() {
    modalEl.classList.remove("hidden");
    document.body.classList.add("modal-open");
    setModalLoadingState("Loading investigation...", true);
  }

  function renderAgentTrace(agentTrace) {
    if (!modalAgentTrace) return;
    if (!Array.isArray(agentTrace) || !agentTrace.length) {
      modalAgentTrace.classList.add("hidden");
      modalAgentTrace.innerHTML = "";
      return;
    }
    const chips = agentTrace
      .map((step) => {
        const label = String(step.agent || "agent")
          .replaceAll("_", " ")
          .replace(/\b\w/g, (c) => c.toUpperCase());
        return agentStatusPill(label, step.status || "completed");
      })
      .join("");
    modalAgentTrace.classList.remove("hidden");
    modalAgentTrace.innerHTML = `
      <div class="text-[11px] font-semibold text-text mb-1">Analysis completed by</div>
      <div class="flex flex-wrap gap-2">${chips}</div>
    `;
  }

  function humanizeAgent(agent) {
    return String(agent || "agent")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function normalizeAgentStatus(status) {
    const token = String(status || "").toLowerCase();
    if (token === "completed" || token === "success" || token === "done") return "completed";
    if (token === "failed" || token === "error") return "failed";
    if (token === "running" || token === "started" || token === "active" || token === "pending") return "running";
    return "running";
  }

  function agentStatusPill(name, status) {
    const normalizedStatus = normalizeAgentStatus(status);
    const styleByStatus = {
      completed: {
        cls: "bg-[#E6F6EC] border-[#2ECC71] text-[#145A32]",
        icon: "✓",
      },
      running: {
        cls: "bg-[#FFF5E6] border-[#F39C12] text-[#8A4B00]",
        icon: "⟳",
      },
      failed: {
        cls: "bg-[#FDECEA] border-[#E74C3C] text-[#7A1F1F]",
        icon: "⚠",
      },
    };
    const selected = styleByStatus[normalizedStatus] || styleByStatus.running;
    return `<span class="inline-flex items-center px-2.5 py-1 rounded-full border text-[11px] font-semibold ${selected.cls}">${selected.icon} ${name}</span>`;
  }

  function initializeAgentProgressUI() {
    const seed = AGENT_SEQUENCE.map((agent) => ({ agent, status: "pending", summary: "" }));
    renderAgentProgressUI(seed);
  }

  function renderAgentProgressUI(rows) {
    if (!modalAgentTrace) return;
    if (!Array.isArray(rows) || !rows.length) {
      modalAgentTrace.classList.add("hidden");
      modalAgentTrace.innerHTML = "";
      return;
    }
    const line = rows.map((r) => humanizeAgent(r.agent)).join(" \u2192 ");
    const chips = rows
      .map((r) => {
        return agentStatusPill(humanizeAgent(r.agent), r.status);
      })
      .join("");
    modalAgentTrace.classList.remove("hidden");
    modalAgentTrace.innerHTML = `
      <div class="text-[11px] font-semibold text-text mb-1">Analysis completed by</div>
      <div class="text-[11px] text-secondaryText mb-2">${line}</div>
      <div class="flex flex-wrap gap-2">${chips}</div>
    `;
  }

  function applyProgressEvents(eventRows, event) {
    if (!event || !event.agent) return;
    const idx = eventRows.findIndex((r) => r.agent === event.agent);
    if (idx === -1) return;
    eventRows[idx] = {
      ...eventRows[idx],
      status: event.status || eventRows[idx].status,
      summary: event.summary || eventRows[idx].summary,
    };
    renderAgentProgressUI(eventRows);
  }

  function setAlertState(alertId, updates) {
    const current = alertsById.get(alertId);
    if (!current) return;
    alertsById.set(alertId, { ...current, ...updates });
    renderAlerts();
  }

  async function openCachedOrStoredInvestigation(alertId) {
    const alertItem = alertsById.get(alertId);
    if (!alertItem) return;
    const cached = investigationsById.get(alertId);
    if (cached) {
      modalEl.classList.remove("hidden");
      document.body.classList.add("modal-open");
      populateModal(alertItem, cached);
      return;
    }

    openModalLoading();
    try {
      const url = `${API_BASE}/investigation/${encodeURIComponent(alertId)}`;
      const resp = await fetchJsonWithStatus(url);
      if (!resp.json || resp.json.ok !== true || !resp.json.data) {
        const backendError = resp.json?.error || "Investigation not available";
        const backendDetails = resp.json?.details || "No details provided";
        setModalErrorState(`${backendError} — ${backendDetails}`);
        return;
      }
      investigationsById.set(alertId, resp.json.data);
      const mappedRecommendation = normalizeRecommendation(
        resp.json.data?.recommendation || resp.json.data?.decision?.recommendation
      );
      const mappedWorkflow = resp.json.data?.workflow_status || "In Review";
      setAlertState(alertId, {
        investigation_status: "investigated",
        has_cached_investigation: true,
        recommendation: mappedRecommendation,
        workflow_status: mappedWorkflow,
        queue_status: mappedRecommendation,
        status: mappedWorkflow,
      });
      populateModal(alertItem, resp.json.data);
    } catch (e) {
      console.error("[UI] open completed investigation failed", e);
      setModalErrorState("Failed to load the completed investigation.");
    }
  }

  async function runInvestigationForAlert(alertId) {
    const alertItem = alertsById.get(alertId);
    if (!alertItem) return;
    setAlertState(alertId, { investigation_status: "investigating", workflow_status: "In Review", recommendation: "Escalate", queue_status: "Escalate" });
    openModalLoadingForInvestigation();
    initializeAgentProgressUI();

    try {
      const txId = encodeURIComponent(alertItem.transaction_id || alertItem.alert_id || alertId);
      const startUrl = `${API_BASE}/investigate/${txId}/start`;
      const startResp = await fetchJsonWithStatus(startUrl, { method: "POST" });
      if (!startResp.json || startResp.json.ok !== true) {
        throw new Error(startResp.json?.error || "Failed to start investigation progress");
      }

      const progressRows = AGENT_SEQUENCE.map((agent) => ({ agent, status: "pending", summary: "" }));
      let seenEvents = 0;
      let result = null;
      while (true) {
        const progressUrl = `${API_BASE}/investigate/${txId}/progress`;
        const progressResp = await fetchJsonWithStatus(progressUrl, { method: "GET" });
        const body = progressResp.json || {};
        const events = Array.isArray(body.events) ? body.events : [];
        const newEvents = events.slice(seenEvents);
        seenEvents = events.length;
        newEvents.forEach((ev) => applyProgressEvents(progressRows, ev));

        if (body.status === "completed" && body.data) {
          result = body.data;
          break;
        }
        if (body.status === "failed") {
          const backendError = body.error || "Unknown backend error";
          const backendDetails = body.details || "No details provided";
          setAlertState(alertId, { investigation_status: "failed", workflow_status: "In Review", recommendation: "Escalate", queue_status: "Escalate" });
          setModalErrorState(`${backendError} — ${backendDetails}`);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 350));
      }

      if (!result) {
        throw new Error("Investigation completed without result data");
      }
      console.log("[UI] investigation response received", result);
      console.log("[UI] agent_trace:", result.agent_trace || []);
      const completedAt = result.investigation_completed_at || new Date().toISOString();
      investigationsById.set(alertId, result);
      const finalRisk = Number(result.risk_score ?? alertItem.risk_score ?? 0);
      const recommendation = normalizeRecommendation(result.recommendation || result.decision?.recommendation);
      const workflowStatus = result.workflow_status || "In Review";
      setAlertState(alertId, {
        investigation_status: "investigated",
        investigation_completed_at: completedAt,
        timestamp: completedAt,
        investigation_summary: result.case_summary?.typology || result.decision?.decision_rationale || null,
        final_risk_score: finalRisk,
        final_recommendation: recommendation,
        has_cached_investigation: true,
        recommendation,
        workflow_status: workflowStatus,
        queue_status: recommendation,
        status: workflowStatus,
        risk_score: finalRisk,
        risk_level: deriveRiskLevel(finalRisk),
      });
      populateModal(alertsById.get(alertId) || alertItem, result);
    } catch (e) {
      console.error("[UI] investigation failed", e);
      setAlertState(alertId, { investigation_status: "failed", workflow_status: "In Review", recommendation: "Escalate", queue_status: "Escalate" });
      if (DEV_FALLBACK_MODAL) {
        populateModal(alertItem, {
          case_summary: {
            alert_trigger: alertItem.alert_type || "Fraud Alert",
            typology: "Fallback Typology",
            transaction_amount: alertItem.amount || 0,
            deviation: `${Math.round((alertItem.risk_score || 0) * 100)}% risk score`,
            recommendation: "Fallback recommendation enabled in dev mode.",
          },
          device_identity: { device_type: "-", device_id: "-", ip_address: "-", country: "-" },
          signals: [],
          supporting_transactions: [],
          beneficiary_analysis: {
            beneficiary_id: alertItem.beneficiary_id || "-",
            fraud_link: false,
            connections: null,
            network_risk_label: "Network Risk",
          },
          decision: {
            risk_score: alertItem.risk_score || 0,
            typology: "Fallback Typology",
            recommendation: "Fallback recommendation enabled in dev mode.",
          },
        });
      } else {
        setModalErrorState("Failed to load investigation details from the API.");
      }
    }
  }

  function populateModal(alertItem, result) {
    stopRecommendationLoadingUI();
    const inv = result.investigation || {};
    const decision = result.decision || {};
    const caseSummary = result.case_summary || {};
    const deviceIdentity = result.device_identity || {};
    const signals = normalizeSignalsPayload(result.signals, inv.base_transaction || {}, result.evidence || []);
    const supporting = Array.isArray(result.supporting_transactions) ? result.supporting_transactions : [];
    const beneficiary = result.beneficiary_analysis || {};
    renderAgentTrace(result.agent_trace);
    if (modalLlmDebug) {
      if (DEV_DEBUG_UI) {
        const provider = result.llm_provider_used || "unknown";
        const model = result.model_used || "unknown";
        modalLlmDebug.classList.remove("hidden");
        modalLlmDebug.textContent = `LLM provider: ${provider} | Model: ${model}`;
      } else {
        modalLlmDebug.classList.add("hidden");
        modalLlmDebug.textContent = "";
      }
    }

    const riskScore = result.risk_score ?? decision.risk_score ?? inv.risk_score ?? alertItem.risk_score ?? 0;
    const riskBand = getRiskBand(riskScore);
    const resolvedRecommendation = normalizeRecommendation(result.recommendation || decision.recommendation);
    const plan = recommendationPlan(resolvedRecommendation);
    const riskLabel = `${riskBand} RISK`;
    const riskBadgeClass = mapRiskLevelToBadgeClass(riskLabel);

    if (modalTitle) modalTitle.textContent = `Alert #${result.alert_id || alertItem.transaction_id}`;
    if (modalRiskPill) {
      modalRiskPill.textContent = `${riskLabel} ${riskScore.toFixed(2)}`;
      modalRiskPill.className = `text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wide ${riskBadgeClass}`;
    }
    applyPrimaryActionUI(plan.recommendation, plan);

    if (modalMeta) {
      const modalTimestamp = result.investigation_completed_at || result.timestamp || alertItem.timestamp;
      const workflowStatus = result.workflow_status || alertItem.workflow_status || "In Review";
      const ts = formatTimestampStacked(modalTimestamp);
      modalMeta.innerHTML = `
        <span class="meta-chip inline-flex items-start gap-2"><i class="fa-regular fa-user mt-0.5"></i><span>${displayOrNA(result.customer_id || inv.account_id || alertItem.account_id)}</span></span>
        <span class="meta-chip inline-flex items-start gap-2"><i class="fa-solid fa-wallet mt-0.5"></i><span>${displayOrNA(alertItem.beneficiary_id)}</span></span>
        <span class="meta-chip inline-flex items-start gap-2"><i class="fa-regular fa-clock mt-0.5"></i><span class="leading-tight"><div>${ts.date}</div><div>${ts.time}</div></span></span>
        <span class="meta-chip inline-flex items-start gap-2"><i class="fa-regular fa-folder-open mt-0.5"></i><span>${displayOrNA(workflowStatus)}</span></span>
      `;
    }

    if (modalAlertTrigger) modalAlertTrigger.textContent = caseSummary.alert_trigger || alertItem.alert_type || "Fraud Alert";
    const resolvedTypology =
      caseSummary.typology ||
      result.fraud_typology ||
      decision.fraud_typology ||
      decision.typology ||
      inv.fraud_typology ||
      inv.typology ||
      "Unknown / Mixed Pattern";
    if (modalTypologyPill) modalTypologyPill.textContent = resolvedTypology;
    if (modalTypologyTooltip) modalTypologyTooltip.textContent = typologyDefinitionFor(result, decision, inv, resolvedTypology);
    if (modalTypologySecondary) {
      const secondary = caseSummary.secondary_typology || "";
      modalTypologySecondary.textContent = secondary;
      if (secondary) modalTypologySecondary.classList.remove("hidden");
      else modalTypologySecondary.classList.add("hidden");
    }
    if (modalAmount) modalAmount.textContent = formatCurrency(caseSummary.transaction_amount ?? alertItem.amount);
    if (modalDeviation) modalDeviation.innerHTML = `<i class="fa-solid fa-arrow-trend-up"></i> ${caseSummary.deviation || `${(riskScore * 100).toFixed(0)}% risk score`}`;
    if (modalRecommendation) {
      const recommendationText = result.recommendation || decision.recommendation || caseSummary.recommendation || plan.recommendation;
      const reasons = deriveReasonBullets(result, caseSummary, decision);
      const nextActions = deriveRecommendedActions(result, decision, recommendationText);
      const recommendationBadgeClass = statusBadgeClasses(plan.recommendation);
      const reasonsHtml = reasons.length
        ? `<ul class="mt-2 list-disc pl-5 space-y-1 text-sm text-blue-900">${reasons
            .map((d) => `<li>${d}</li>`)
            .join("")}</ul>`
        : "";
      const actionsHtml = nextActions.length
        ? `<ol class="mt-2 list-decimal pl-5 space-y-1 text-sm text-text">${nextActions
            .map((step) => `<li>${step}</li>`)
            .join("")}</ol>`
        : "";
      modalRecommendation.innerHTML = `
        <div class="text-xs uppercase tracking-wide text-secondaryText font-semibold mb-1">Recommendation</div>
        <div class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold border mb-2 ${recommendationBadgeClass}">${plan.recommendation}</div>
        <div class="text-sm text-text">${recommendationText}</div>
        <div class="text-xs uppercase tracking-wide text-secondaryText font-semibold mt-3 mb-1">Reason</div>
        ${reasonsHtml}
        <div class="text-xs uppercase tracking-wide text-secondaryText font-semibold mt-4 mb-1">Recommended Next Actions</div>
        ${actionsHtml}
      `;
    }

    const baseTx = inv.base_transaction || {};
    if (modalDeviceType) modalDeviceType.textContent = displayOrNA(deviceIdentity.device_type || baseTx.device_type || baseTx.device_id);
    if (modalLocation) modalLocation.textContent = displayOrNA(deviceIdentity.location || deviceIdentity.country);
    if (modalDeviceFingerprint) modalDeviceFingerprint.textContent = displayOrNA(deviceIdentity.device_fingerprint || baseTx.device_id);
    if (modalIpAddress) modalIpAddress.textContent = displayOrNA(deviceIdentity.ip_address || baseTx.ip_address);

    renderSignalCards(signals);

    if (modalSupportingTbody) {
      modalSupportingTbody.innerHTML = "";
      const rows = supporting.length
        ? supporting
        : [
            {
              time: alertItem.timestamp,
              destination_account: alertItem.beneficiary_id || inv.account_id || "-",
              amount: alertItem.amount || 0,
              balance_after: null,
            },
          ];
      rows.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="px-4 py-2 text-text">${formatTimestamp(r.time)}</td>
          <td class="px-4 py-2 font-mono text-secondaryText">${r.destination_account || "-"}</td>
          <td class="px-4 py-2 font-medium text-text">${formatCurrency(r.amount || 0)}</td>
          <td class="px-4 py-2 text-secondaryText">${r.balance_after ?? "-"}</td>
        `;
        modalSupportingTbody.appendChild(tr);
      });
    }

    if (modalNetworkTbody) {
      modalNetworkTbody.innerHTML = "";
      const beneficiaryId = beneficiary.beneficiary_id || alertItem.beneficiary_id || "-";
      const row = document.createElement("tr");
      row.className = beneficiary.fraud_link ? "bg-red-50/50" : beneficiary.suspected_network_risk ? "bg-amber-50/50" : "";
      const linkText = beneficiary.fraud_link
        ? "Yes (Confirmed)"
        : beneficiary.suspected_network_risk
        ? "Suspected"
        : "No";
      const linkClass = beneficiary.fraud_link
        ? "text-highRisk font-bold"
        : beneficiary.suspected_network_risk
        ? "text-amber-700 font-semibold"
        : "text-secondaryText";
      row.innerHTML = `
        <td class="px-4 py-2 font-mono font-medium text-primary">${beneficiaryId}</td>
        <td class="px-4 py-2 text-text">Beneficiary</td>
        <td class="px-4 py-2 ${linkClass}">${linkText}</td>
        <td class="px-4 py-2 text-text">${beneficiary.connections ?? "-"}</td>
      `;
      modalNetworkTbody.appendChild(row);
      if (modalNetworkRiskPill) {
        const networkLabel = beneficiary.network_risk_label || "Low Network Risk";
        modalNetworkRiskPill.textContent = networkLabel;
        modalNetworkRiskPill.className = `text-xs px-2 py-0.5 rounded border ${mapRiskLevelToBadgeClass(networkLabel)}`;
      }
    }
    renderBeneficiaryGraph(beneficiary, result.customer_id || inv.account_id || "You");
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindTableDelegation();
    loadMetrics();
    loadAlerts();
  });
})();

