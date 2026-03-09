## Agentic Fraud Investigation Co-Pilot – MVP

This project implements an agentic fraud investigation co-pilot as described in the PRD. It ingests synthetic fraud datasets, builds enriched transaction context, runs a bounded-autonomy CrewAI-style investigation pipeline, and exposes results via a FastAPI backend and a Tailwind-based dashboard frontend.

### 1. Project structure

- `app.py` – FastAPI entrypoint and API routes
- `models/` – Pydantic models (`EnrichedTransactionRecord`, `EvidenceItem`, `InvestigationState`, etc.)
- `services/` – core services:
  - `data_repository.py` – dataset loading and standardisation
  - `investigation_feature_builder.py` – builds `EnrichedTransactionRecord`
  - `evidence_registry.py` – collects structured evidence
  - `graph_builder.py` – NetworkX account graph utilities
  - `scoring_engine.py` – deterministic risk scoring
  - `investigation_pipeline.py` – orchestrates end-to-end investigations
- `agents/` – bounded-autonomy investigation agents (Pattern, Behaviour, Network, Typology, Decision)
- `frontend/` – dashboard UI (`index.html`, `app.js`)
- `data/` – CSV datasets (PaySim, IEEE)
- `tests/` – pytest test suite for models, services, agents, pipeline, and API

### 2. Environment and `.env`

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-key-here
```

The key is **never** hardcoded; it is loaded via `python-dotenv` in `app.py` and in the agent base class. The current MVP agents are deterministic and do not require the key to run, but the key is available for future CrewAI/LLM integration.

### 3. Setup and installation

Requirement: Python 3.12 with a working SSL setup and access to PyPI.

```bash
cd "Fraud Co-Pilot MVP"

python3 -m venv .venv
source .venv/bin/activate   # On macOS/Linux

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` file pins `numpy==1.26.4` and includes FastAPI, pandas, NetworkX, CrewAI, and pytest.

### 4. Running the backend locally

With the virtualenv activated:

```bash
uvicorn app:app --reload --port 7860
```

This starts FastAPI on `http://127.0.0.1:7860` with the following key endpoints:

- `GET /health` – health probe
- `POST /investigate` – run an investigation for a given `transaction_id`
- `GET /alerts` – dashboard alerts queue
- `GET /metrics` – dashboard KPI metrics

### 5. Running the frontend dashboard

Open the static dashboard in a browser pointing at the same origin:

```bash
cd frontend
python -m http.server 8000
```

Then visit `http://127.0.0.1:8000/index.html`.

If the API is on a different origin, set `window.API_BASE` in `index.html` or via a small inline script before loading `app.js`:

```html
<script>window.API_BASE = "http://127.0.0.1:7860";</script>
<script src="./app.js"></script>
```

The dashboard will:

- Populate KPI cards from `GET /metrics`
- Populate the alerts table from `GET /alerts`
- Trigger `POST /investigate` when clicking **Investigate** and fill the investigation modal

### 6. Running tests

With the virtualenv activated:

```bash
pytest
```

The suite covers:

- Core models (`tests/test_models.py`)
- Data repository and feature builder (`tests/test_data_repository.py`, `tests/test_feature_builder.py`)
- Evidence registry and scoring engine (`tests/test_evidence_registry.py`, `tests/test_scoring_engine.py`)
- Graph builder (`tests/test_graph_builder.py`)
- Agents (`tests/test_agents.py`)
- Investigation pipeline (`tests/test_pipeline.py`)
- FastAPI API layer (`tests/test_api.py`)

> **Note:** In some environments (including this coding sandbox), `pytest` may fail with a low-level `numpy` / OpenBLAS segmentation fault when loading large CSVs. If you see such errors, verify your local Python/numpy installation (or re-install Python via a package manager such as Homebrew) and re-run the tests.

### 7. Docker deployment

Build and run the container:

```bash
docker build -t fraud-copilot .
docker run --env-file .env -p 7860:7860 fraud-copilot
```

This uses `uvicorn` to serve `app:app` on port `7860` inside the container, matching the PRD requirement.

### 8. Remaining gaps and notes

- **CrewAI integration:** The current agents are implemented as deterministic Python classes with a stable JSON contract and bounded autonomy. They are ready to be wrapped into CrewAI `Agent` objects and a sequential `Crew` but do not call the OpenAI API yet.
- **Dataset size:** For local testing, you may want to adjust `DataRepositoryConfig(max_rows=...)` to limit CSV reads.
- **Environment-specific issues:** If you encounter `numpy`-related segmentation faults when running tests, this is almost always due to a broken or mismatched system BLAS/`numpy` installation rather than the project code. Recreating the virtualenv or reinstalling Python typically resolves it.

