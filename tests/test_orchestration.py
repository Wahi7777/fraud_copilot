from pathlib import Path

from crewai import Crew, Process

from agents.behaviour_agent import BehaviourAgent
from agents.decision_agent import DecisionAgent
from agents.network_agent import NetworkAgent
from agents.pattern_agent import PatternAgent
from agents.typology_agent import TypologyAgent
from orchestration.crew import InvestigationCrew
from services.data_repository import DataRepository, DataRepositoryConfig
from services.graph_builder import GraphBuilder
from services.llm_provider import get_llm


def _minimal_graph_builder() -> GraphBuilder:
  # Use a small repository with a tight max_rows to avoid heavy loads.
  repo = DataRepository.load_from_csvs(DataRepositoryConfig(data_dir=Path("data"), max_rows=10))
  return GraphBuilder(repo)


def test_investigation_crew_uses_sequential_process_and_crewai_objects():
  gb = _minimal_graph_builder()
  pattern = PatternAgent()
  behaviour = BehaviourAgent()
  network = NetworkAgent(gb)
  typology = TypologyAgent()
  decision = DecisionAgent()

  crew = InvestigationCrew(
      pattern_agent=pattern,
      behaviour_agent=behaviour,
      network_agent=network,
      typology_agent=typology,
      decision_agent=decision,
  )

  # Validate CrewAI primitives are present and configured as expected.
  assert isinstance(crew.crew, Crew)
  assert crew.process == Process.sequential
  assert len(crew.tasks) == 5
  assert len(crew.agents) == 5

  # Each CrewAI agent must have an LLM attached.
  llm = get_llm()
  assert crew.pattern_agent.llm is not None
  assert crew.behaviour_agent.llm is not None
  assert crew.network_agent.llm is not None
  assert crew.typology_agent.llm is not None
  assert crew.decision_agent.llm is not None

