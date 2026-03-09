from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dotenv import load_dotenv

from models import EnrichedTransactionRecord, EvidenceItem

# Ensure .env is loaded so OPENAI_API_KEY is available when CrewAI is used.
load_dotenv()


class BaseInvestigationAgent(ABC):
    """
    Base class for bounded-autonomy investigation agents.

    Agents implement deterministic domain logic and may optionally be wrapped
    as CrewAI agents elsewhere. All `run` methods must return structured JSON.
    """

    def __init__(self, name: str, description: str, domain: str) -> None:
        self.name = name
        self.description = description
        self.domain = domain

    @abstractmethod
    def run(self, tx: EnrichedTransactionRecord, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute this agent's analysis for a single transaction and return a
        JSON-serialisable dictionary.
        """

    @staticmethod
    def _serialize_evidence(items: List[EvidenceItem]) -> List[Dict[str, Any]]:
        return [e.model_dump() for e in items]

