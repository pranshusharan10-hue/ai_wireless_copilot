# AI-Wireless Copilot

**LLM-Assisted Wireless Communication Simulation and Optimization**

## Overview

AI-Wireless Copilot is a domain-specific AI engineering project that translates natural-language wireless communication requests into validated simulation configurations, executes trusted wireless simulations, retrieves relevant wireless-domain knowledge, and returns quantitative results with engineering explanations.

The project combines wireless communication theory, Python-based simulation, large language models (LLMs), retrieval-augmented generation (RAG), API development, scientific validation, and constrained optimization.

## Current Status

**Current milestone: Phase 3 — LLM Integration and Wireless-Domain RAG**

Implemented and tested capabilities include:

- BPSK and QPSK modulation and demodulation.
- BER-versus-Eb/N0 simulations over AWGN, Rayleigh, and Rician channels.
- Theoretical BER calculations and comparison with simulated results.
- Configurable simulation parameters, random seeds, and Rician K-factor.
- FastAPI endpoints for request validation, simulation, and natural-language copilot requests.
- Local LLM integration through Ollama for extracting structured simulation configurations.
- Wireless-domain knowledge retrieval using Ollama embeddings and cosine similarity.
- Pydantic-based configuration validation before simulation execution.
- Compact copilot responses by default, with optional constellation samples.
- Automated tests for the simulation engine, API, configuration validation, RAG, and copilot workflow.

Latest verified baseline: **46 tests passed**, Ruff checks passed, and the Starlette test-client deprecation warning was resolved in the development environment.

## Technology Stack

- **Language:** Python 3.11
- **Simulation and numerical computing:** NumPy, SciPy
- **Visualization:** Matplotlib
- **API:** FastAPI, Uvicorn
- **Data validation:** Pydantic
- **LLM runtime:** Ollama
- **Local LLM:** `qwen2.5:3b`
- **Embedding model:** `nomic-embed-text`
- **Retrieval:** Embeddings and cosine similarity over a Markdown knowledge base
- **Testing:** Pytest, HTTPX, HTTPX2
- **Code quality:** Ruff
- **Planned interface:** Streamlit

## Architecture

The current copilot workflow is:

1. Receive a natural-language simulation request.
2. Retrieve relevant wireless-domain knowledge.
3. Ask the LLM to extract a structured simulation configuration.
4. Validate the configuration using the project's trusted validation logic.
5. Execute the simulation using the existing Python simulation engine.
6. Return the configuration, simulation metrics, retrieved knowledge, and a short explanation.

The LLM does not directly execute generated Python code. It proposes configuration values; the application validates those values and passes them to the trusted simulation engine.

## Development Roadmap

- [x] **Phase 0 — Environment and Project Foundation:** Repository structure, development environment, API foundation, and test setup.
- [x] **Phase 1 — Wireless Simulation Foundation:** BPSK/QPSK simulation over AWGN, modulation and demodulation, BER calculation, and simulation API.
- [x] **Phase 2 — Channel Modeling and Scientific Validation:** Rayleigh/Rician fading, channel equalization, theoretical BER comparisons, and regression tests.
- [x] **Phase 3 — LLM Integration and Wireless-Domain RAG:** Ollama-based configuration extraction, knowledge retrieval, validation, copilot API integration, and compact responses.
- [ ] **Phase 4 — Controlled Execution and Automatic Correction:** Improve failure diagnosis and implement bounded, validated correction workflows.
- [ ] **Phase 5 — Wireless Optimization:** Explore adaptive modulation and parameter selection under BER or other communication constraints.
- [ ] **Phase 6 — Optional Fine-Tuning and Model Evaluation:** Evaluate whether fine-tuning or parameter-efficient methods such as QLoRA improve configuration extraction and domain-specific performance.
- [ ] **Phase 7 — Benchmarking, Interface, and Final Documentation:** Build evaluation datasets, compare models, develop the user interface, and prepare final project documentation.

A phase is considered complete only after its implementation, tests, documentation, and Git changes have been reviewed.

## Prerequisites

- Python 3.11
- Git
- Ollama installed and available from the terminal
- Sufficient system memory to run the selected local models

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai_wireless_copilot
```

Replace `<repository-url>` with the actual Git repository URL.

### 2. Create and activate the Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, use an appropriate execution-policy setting for your environment or activate the environment through your IDE.

### 3. Download the Ollama models

Make sure Ollama is installed and running, then execute:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

The first model extracts simulation configuration from natural language. The second generates embeddings for wireless-domain retrieval.

### 4. Start the API

From the project root, with the virtual environment activated:

```powershell
python -m uvicorn backend.main:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

Check that the service is running using:

`http://127.0.0.1:8000/health`

## API Usage

### Validate a simulation request

Endpoint: `POST /api/v1/simulations/validate`

Example:

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

### Run a direct simulation

Endpoint: `POST /api/v1/simulations/run`

This endpoint runs a simulation from an explicitly supplied configuration. The currently supported modulation schemes for execution are BPSK and QPSK.

### Run the AI copilot

Endpoint: `POST /api/v1/copilot/run`

Example:

```json
{
  "prompt": "Simulate BPSK over Rayleigh fading from 0 to 10 dB in steps of 2 dB using 10000 bits"
}
```

The copilot extracts and validates the configuration, retrieves relevant knowledge, runs the simulation, and returns simulated and theoretical BER values.

By default, constellation samples are omitted to keep the response compact.

To include constellation samples:

```json
{
  "prompt": "Simulate BPSK over Rayleigh fading from 0 to 10 dB in steps of 2 dB using 10000 bits",
  "include_constellation": true
}
```

## Testing and Code Quality

Run the automated test suite:

```powershell
python -m pytest
```

Run Ruff lint checks:

```powershell
python -m ruff check .
```

Check code formatting:

```powershell
python -m ruff format --check .
```

Apply automatic formatting when required:

```powershell
python -m ruff format .
```

The latest verified development baseline is 46 passing tests. Re-run these checks after modifying code or dependencies.

## Phase Completion Protocol

At the end of every development phase, follow this checklist before starting the next phase:

1. **Implementation:** Confirm that the planned features for the phase are implemented.
2. **Validation:** Run relevant functional checks and regression tests.
3. **Code quality:** Run Ruff lint and formatting checks.
4. **Documentation:** Update this README with implemented capabilities, API changes, dependencies, setup changes, test results, and remaining limitations.
5. **Review:** Inspect `git diff` and `git diff --check`; verify that no secrets or local environment files are included.
6. **Version control:** Commit the implementation, tests, and documentation together using a descriptive commit message.
7. **Roadmap:** Mark the phase complete only after its acceptance criteria have been met.

Do not mark future or partially implemented capabilities as completed.

## Reproducibility

Simulation requests support explicit random seeds, configurable SNR ranges, and configurable bit counts. The API returns simulation parameters and BER results so that experiments can be reproduced and compared.

Record dependency changes in `requirements.txt`. Keep model names, runtime configuration, test commands, and known limitations documented as the project evolves.

## Safety and Limitations

- Natural-language requests are converted into structured configurations rather than executable code.
- Configuration values are validated before the trusted simulation engine runs.
- Unsupported modulation schemes must not bypass validation.
- RAG results are reference material and must not override explicit user parameters.

### Phase 4 — Controlled Execution and Automatic Correction

**Objective:** Improve the reliability of the AI copilot by evaluating LLM configuration extraction, handling failures consistently, and introducing bounded automatic correction.

#### Phase 4.1A — Benchmark Dataset
- [x] Create a benchmark dataset containing valid and invalid wireless simulation prompts.
- [x] Define expected configurations for each benchmark case.
- [x] Add dataset structure and uniqueness checks.
- [x] Verify valid and invalid configurations using the existing configuration validator.

#### Phase 4.2 — Error Classification and Handling
- [x] Separate RAG retrieval, LLM extraction, configuration validation, and simulation execution errors.
- [x] Return HTTP 422 for invalid simulation configurations.
- [x] Return HTTP 503 for LLM and RAG service failures.
- [x] Return HTTP 500 for unexpected simulation execution failures.
- [x] Add server-side exception logging and safe API error messages.
- [x] Add regression tests for RAG failures and invalid configurations reaching the simulator.

#### Phase 4.1B — Actual LLM Benchmark Evaluation
- [ ] Evaluate the actual local Ollama model against the benchmark prompts.
- [ ] Compare extracted configurations against expected configurations.
- [ ] Calculate exact-match and field-level accuracy.
- [ ] Measure validation success, invalid-output rate, and extraction failures.
- [ ] Generate a machine-readable evaluation report.

#### Phase 4.3 — Bounded Automatic Correction
- [ ] Implement a maximum of one correction attempt for invalid LLM configurations.
- [ ] Revalidate corrected configurations before simulation.
- [ ] Prevent infinite correction loops.
- [ ] Ensure invalid configurations never reach the simulation engine.

#### Phase 4.4 — Safety and Regression Testing
- [ ] Test malformed LLM responses and unexpected service failures.
- [ ] Verify correction behavior and retry limits.
- [ ] Run the complete regression test suite.
- [ ] Run Ruff lint and formatting checks.

#### Phase 4.5 — Documentation and Release
- [ ] Update README with implementation details, API behavior, evaluation results, and limitations.
- [ ] Review changes and verify no secrets or unrelated files are committed.
- [ ] Commit and push the completed Phase 4 implementation.

**Phase 4 completion criteria:** All implementation, evaluation, safety, regression, documentation, and Git acceptance checks must pass.