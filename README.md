# AI-Wireless Copilot

**LLM-Assisted Wireless Communication Simulation and Optimization**

AI-Wireless Copilot is a domain-specific AI engineering project that converts natural-language wireless communication requests into structured simulation configurations, validates those configurations, executes trusted wireless simulations, and returns quantitative results with relevant wireless-domain knowledge.

The project brings together wireless communication theory, Python-based simulation, local large language models (LLMs), retrieval-augmented generation (RAG), API development, scientific validation, and controlled LLM evaluation.

---

## Project Status

**Current milestone: Phase 4 â€” Controlled Execution, Evaluation, and Safety**

**Current activity: Phase 4.5 â€” Documentation and release preparation**

Implemented capabilities:

- BPSK and QPSK modulation and demodulation.
- BER-versus-Eb/N0 simulation over AWGN, Rayleigh, and Rician fading channels.
- Theoretical BER calculations and comparison with simulated results.
- Configurable SNR ranges, bit counts, random seeds, and Rician K-factor.
- FastAPI endpoints for request validation, direct simulation, and natural-language copilot requests.
- Local LLM integration through Ollama for extracting structured simulation configurations.
- Wireless-domain knowledge retrieval using Ollama embeddings and cosine similarity.
- Pydantic-based validation before simulation execution.
- Bounded correction: at most one correction attempt for an invalid LLM-generated configuration.
- Defensive handling of malformed responses and LLM, RAG, validation, and simulation failures.
- Security regression tests for untrusted prompts, retrieved context, malformed configurations, and safe error responses.
- Development and held-out benchmark datasets for evaluating LLM configuration extraction.
- Machine-readable benchmark reports.
- Compact copilot responses by default, with optional constellation samples.

**Latest recorded verification:** 102 tests passed. Ruff lint and formatting checks passed, and `git diff --check` reported no whitespace errors. Re-run the checks locally before treating this as the current result for a new code revision.

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| Simulation and numerical computing | NumPy, SciPy |
| Visualization | Matplotlib |
| API | FastAPI, Uvicorn |
| Data validation | Pydantic |
| LLM runtime | Ollama |
| Local LLM | `qwen2.5:3b` |
| Embedding model | `nomic-embed-text` |
| Retrieval | Embeddings and cosine similarity |
| Knowledge base | Markdown |
| Testing | Pytest, HTTPX |
| Code quality | Ruff |

The project currently exposes its functionality through a FastAPI backend. A Streamlit interface is not listed as an implemented capability.

---

## Architecture

The copilot workflow is:

1. **Receive a request:** The API receives a natural-language wireless simulation request.
2. **Retrieve domain knowledge:** The RAG component retrieves relevant material from the wireless knowledge base.
3. **Extract configuration:** The local Ollama model converts the request into structured simulation parameters.
4. **Validate configuration:** Trusted application validation checks the extracted values and rejects unsupported or invalid configurations.
5. **Correct boundedly, if needed:** An invalid LLM-generated configuration can trigger at most one correction attempt. The corrected result is validated again.
6. **Execute trusted simulation:** The validated configuration is passed to the existing Python simulation engine.
7. **Return results:** The API returns the configuration, simulation metrics, relevant retrieved context, and an explanation.

**Execution boundary:** The LLM proposes configuration values; it does not execute generated Python code. Retrieved documents and model output are treated as untrusted input. Only validated configuration values are passed to the trusted simulation implementation.

---

## Supported Simulation Scope

### Modulation

- BPSK
- QPSK

### Channel models

- AWGN
- Rayleigh fading
- Rician fading

### Simulation outputs

- Simulated bit error rate (BER)
- Theoretical BER where supported by the implementation
- BER-versus-Eb/N0 results
- Simulation configuration and error counts
- Optional received constellation samples

Although the request schema may contain additional modulation labels, the current simulation execution path supports **BPSK and QPSK only**. Do not assume that 16QAM or 64QAM simulation is implemented.

---

## Project Structure

```text
ai_wireless_copilot/
â”‚
â”œâ”€â”€ app/
â”‚
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ main.py
â”‚   â””â”€â”€ schemas.py
â”‚
â”œâ”€â”€ evaluation/
â”‚   â””â”€â”€ reports/
â”‚       â”œâ”€â”€ development/
â”‚       â””â”€â”€ heldout/
â”‚
â”œâ”€â”€ experiments/
â”‚   â”œâ”€â”€ ber_curve.png
â”‚   â””â”€â”€ qpsk_ber_curve.png
â”‚
â”œâ”€â”€ llm/
â”‚   â”œâ”€â”€ client.py
â”‚   â””â”€â”€ validator.py
â”‚
â”œâ”€â”€ rag/
â”‚   â”œâ”€â”€ knowledge_base/
â”‚   â”‚   â””â”€â”€ wireless_basics.md
â”‚   â””â”€â”€ retriever.py
â”‚
â”œâ”€â”€ runner/
â”‚
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ evaluate_llm.py
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ fixtures/
â”‚   â”‚   â”œâ”€â”€ copilot_benchmark.json
â”‚   â”‚   â””â”€â”€ copilot_heldout.json
â”‚   â”œâ”€â”€ test_api.py
â”‚   â”œâ”€â”€ test_benchmark_dataset.py
â”‚   â”œâ”€â”€ test_copilot.py
â”‚   â”œâ”€â”€ test_evaluate_llm.py
â”‚   â”œâ”€â”€ test_llm_client.py
â”‚   â”œâ”€â”€ test_llm_validator.py
â”‚   â”œâ”€â”€ test_rag.py
â”‚   â””â”€â”€ test_wireless.py
â”‚
â”œâ”€â”€ training/
â”‚
â”œâ”€â”€ validation/
â”‚
â”œâ”€â”€ wireless/
â”‚   â””â”€â”€ simulation.py
â”‚
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

Some directories are currently package or workspace placeholders. The tree above focuses on the main source, evaluation, and test files rather than virtual-environment and cache files.

---

## Prerequisites

- Python 3.11
- Git
- Ollama installed and available from the terminal
- Sufficient system memory to run the selected local models

The default local LLM is `qwen2.5:3b`. The embedding model is `nomic-embed-text`.

---

## Installation and Setup

### 1. Clone the repository

```powershell
git clone https://github.com/pranshusharan10-hue/ai_wireless_copilot.git
cd ai_wireless_copilot
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Confirm that the environment is using Python 3.11:

```powershell
python --version
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Pull the Ollama models

Make sure Ollama is installed and running, then execute:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

The LLM extracts structured simulation configurations from natural-language requests. The embedding model creates vectors used by the wireless-domain retrieval component.

### 5. Start the API

From the project root, with the virtual environment activated:

```powershell
python -m uvicorn backend.main:app --reload
```

The API will be available at:

- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

The copilot endpoint requires the local Ollama service and the configured models to be available.

---

## API Reference

The current API exposes the following main endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/api/v1/simulations/validate` | Validate a supplied simulation configuration |
| POST | `/api/v1/simulations/run` | Run a simulation from a supplied configuration |
| POST | `/api/v1/copilot/run` | Run the natural-language copilot workflow |

### 1. Validate a simulation request

`POST /api/v1/simulations/validate`

Example request:

```json
{
  "prompt": "Test BPSK over AWGN",
  "modulation": "BPSK",
  "channel": "AWGN",
  "snr_start_db": 0,
  "snr_stop_db": 10,
  "snr_step_db": 2,
  "num_bits": 10000,
  "random_seed": 42
}
```

### 2. Run a direct simulation

`POST /api/v1/simulations/run`

This endpoint runs a simulation using the explicitly supplied configuration. The current execution implementation supports BPSK and QPSK over AWGN, Rayleigh, and Rician channels.

### 3. Run the AI copilot

`POST /api/v1/copilot/run`

Example request:

```json
{
  "prompt": "Simulate BPSK over Rayleigh fading from 0 to 10 dB in steps of 2 dB using 10000 bits"
}
```

The copilot retrieves relevant wireless knowledge, asks the local LLM to extract a configuration, validates the result, and runs the simulation if the configuration is valid.

By default, constellation samples are omitted to keep the response compact.

To request constellation samples:

```json
{
  "prompt": "Simulate BPSK over Rayleigh fading from 0 to 10 dB in steps of 2 dB using 10000 bits",
  "include_constellation": true
}
```

For the complete request and response schemas, use the interactive API documentation at `http://127.0.0.1:8000/docs`.

---

## LLM Benchmark Evaluation

The project includes separate development and held-out benchmark datasets for evaluating natural-language-to-configuration extraction.

### Evaluation setup

| Setting | Value |
|---|---|
| Model | `qwen2.5:3b` |
| Runtime | Local Ollama |
| Temperature | 0 |
| Simulation execution | Disabled |
| Required configuration fields | 7 |

### Development benchmark

| Metric | Result |
|---|---:|
| Total cases | 10 |
| Successful extractions | 10/10 |
| Exact configuration matches | 10/10 |
| Field-level accuracy | 100% |
| Validation accuracy | 100% |
| Average latency | 9.74 seconds |
| Timeouts | 0 |

### Held-out benchmark

| Metric | Result |
|---|---:|
| Total cases | 16 |
| Successful extractions | 16/16 |
| Exact configuration matches | 16/16 |
| Field-level accuracy | 100% |
| Validation accuracy | 100% |
| Valid case acceptance | 100% |
| Invalid case rejection | 100% |
| Average latency | 8.63 seconds |
| Median latency | 7.82 seconds |
| Maximum latency | 20.96 seconds |
| Timeouts | 0 |

### Run the evaluations

Run from the project root:

```powershell
python -m scripts.evaluate_llm --dataset development
python -m scripts.evaluate_llm --dataset heldout
```

Machine-readable reports are generated under:

```text
evaluation/reports/development/
evaluation/reports/heldout/
```

**Interpretation:** These results describe the specific development and held-out datasets used in this evaluation. They do not establish 100% accuracy for arbitrary or unseen user prompts. Simulation execution was disabled during these benchmark runs so that extraction and validation performance could be evaluated separately.

---

## Testing and Code Quality

Run the complete automated test suite:

```powershell
python -m pytest
```

Run Ruff lint checks:

```powershell
python -m ruff check .
```

Check formatting:

```powershell
python -m ruff format --check .
```

Apply formatting if needed:

```powershell
python -m ruff format .
```

Check whitespace and patch errors before committing:

```powershell
git diff --check
```

The latest recorded verification baseline is **102 passing tests**, with Ruff lint, Ruff formatting, and `git diff --check` passing. Run these commands again after making changes and update this README if the verified results change.

---

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Environment and project foundation | Complete |
| Phase 1 | BPSK/QPSK simulation over AWGN | Complete |
| Phase 2 | Rayleigh/Rician channel modeling and scientific validation | Complete |
| Phase 3 | Ollama integration, wireless-domain RAG, validation, and copilot API | Complete |
| Phase 4.1A | Benchmark dataset and integrity tests | Complete |
| Phase 4.1B | Actual LLM development and held-out evaluation | Complete |
| Phase 4.2 | Error classification and handling | Complete |
| Phase 4.3 | Bounded automatic correction | Complete |
| Phase 4.4 | Safety and regression testing | Complete |
| Phase 4.5 | Documentation and release preparation | In progress |
| Phase 5 | Wireless optimization | Planned |
| Phase 6 | Optional fine-tuning and model evaluation | Planned |

### Phase 5 â€” Wireless Optimization

Explore adaptive modulation and parameter selection under BER or other communication constraints. This phase is planned and should not be considered implemented.

### Phase 6 â€” Optional Fine-Tuning and Model Evaluation

Evaluate whether fine-tuning or parameter-efficient methods such as QLoRA improve domain-specific configuration extraction. This phase is planned.

---

## Reproducibility

Simulation requests support explicit random seeds, configurable SNR ranges, and configurable bit counts. The API returns simulation parameters and BER results to support repeatable experiments.

For reproducible LLM evaluation, retain the benchmark datasets, model/runtime settings, and generated reports. Model output and latency can depend on the local runtime and hardware environment.

---

## Safety, Validation, and Limitations

- Natural-language requests are converted into structured configuration values rather than executable code.
- LLM output is validated before it can reach the simulation engine.
- At most one bounded correction attempt is made for an invalid extracted configuration.
- Invalid configurations must not reach the simulator.
- Retrieved RAG content is reference material and must not override explicit user parameters.
- Malformed model responses and service failures are handled through controlled error paths.
- The current simulation execution path supports BPSK and QPSK; 16QAM and 64QAM execution is not currently supported.
- Benchmark accuracy is measured on the included datasets and should not be generalized to all possible prompts.
- The copilot depends on a running local Ollama service and the required models.
- Wireless optimization, fine-tuning, and a graphical Streamlit interface are not presented as completed capabilities.

---

## Phase Completion and Release Checklist

Before marking a development phase complete:

1. Confirm that the planned implementation is complete.
2. Run relevant functional checks and regression tests.
3. Run the full test suite.
4. Run Ruff lint and formatting checks.
5. Update documentation for implemented features, API behavior, dependencies, setup, and limitations.
6. Inspect `git diff` and run `git diff --check`.
7. Verify that secrets, local environment files, temporary outputs, and unrelated files are not staged.
8. Commit implementation, tests, and documentation with a descriptive commit message.
9. Push the reviewed changes to the repository.
10. Update the roadmap only after the phase acceptance criteria have been met.

Do not mark planned or partially implemented capabilities as complete.
