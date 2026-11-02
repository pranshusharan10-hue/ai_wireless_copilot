# AI-Wireless Copilot

**LLM-Assisted Wireless Communication Simulation and Optimization**

AI-Wireless Copilot is a wireless communication simulation project that combines Python-based communication models with a local Large Language Model (LLM) and Retrieval-Augmented Generation (RAG).

The project converts natural-language wireless simulation requests into structured configurations, validates the configurations, executes trusted simulation code, and returns quantitative results.

It also contains standalone implementations for 16-QAM and OFDM simulations.

---

## 1. Project Status

**Current milestone:** phase 4
**Current activity:** Phase 4.5 — Documentation and Release Preparation (final verification).

### Implemented capabilities

- BPSK and QPSK modulation and demodulation.
- BER-versus-Eb/N0 simulation over AWGN, Rayleigh, and Rician fading channels.
- Theoretical BER calculations for supported BPSK/QPSK channel combinations.
- Configurable SNR ranges, bit counts, random seeds, and Rician K-factor.
- FastAPI endpoints for health checks, request validation, direct simulation, and natural-language Copilot requests.
- Local LLM integration through Ollama.
- Wireless-domain knowledge retrieval using Ollama embeddings and cosine similarity.
- Pydantic-based validation of extracted configurations.
- At most one bounded correction attempt for an invalid LLM-generated configuration.
- Defensive handling of malformed LLM responses and LLM, RAG, validation, and simulation failures.
- Security regression tests for untrusted prompts, retrieved context, malformed configurations, and safe error handling.
- Development and held-out benchmark datasets for LLM configuration extraction.
- Machine-readable benchmark reports.
- Optional constellation samples in Copilot responses.
- Standalone Gray-coded 16-QAM modulation and demodulation.
- Standalone 16-QAM BER simulation over AWGN, Rayleigh, and Rician channels.
- Standalone OFDM simulation using QPSK and 16-QAM over AWGN and multipath Rayleigh channels.

**Latest verification:** 152 automated tests passed. Ruff linting, Ruff formatting checks, and `git diff --check` passed during Phase 4.5 verification.

---

## 2. Technology Stack

| Area | Technology |
|---|---|
| Programming language | Python 3.11 |
| Numerical computing | NumPy, SciPy |
| Visualization | Matplotlib |
| API framework | FastAPI |
| API server | Uvicorn |
| Data validation | Pydantic |
| LLM runtime | Ollama |
| Default LLM | `qwen2.5:3b` |
| Embedding model | `nomic-embed-text` |
| Retrieval | Embeddings and cosine similarity |
| Testing | Pytest, HTTPX |
| Code quality | Ruff |

The current application interface is the FastAPI backend. A Streamlit graphical interface is not documented as an implemented feature.

---

## 3. System Architecture

### Natural-language Copilot workflow

1. **User request:** The client sends a natural-language wireless simulation request.
2. **Knowledge retrieval:** The RAG component retrieves relevant wireless-domain information.
3. **Configuration extraction:** The local Ollama model converts the request into structured simulation parameters.
4. **Validation:** The trusted validation layer checks the extracted configuration.
5. **Bounded correction:** If an LLM-generated configuration is invalid, the system can request one correction attempt and validate the result again.
6. **Simulation:** The validated configuration is passed to the trusted Python simulation engine.
7. **Response:** The API returns simulation metrics, configuration details, relevant context, and an explanation.

### Execution and security boundary

The LLM proposes configuration values; it does not execute generated Python code.

User prompts, retrieved documents, and model outputs are treated as untrusted input. The application validates configuration values before passing them to the simulation implementation.

Retrieved documents provide reference information and must not override explicit user parameters.

---

## 4. Simulation Capabilities

The repository contains an integrated Copilot simulation path and additional standalone wireless simulation modules. These are documented separately because they are not all exposed through the same API workflow.

### 4.1 Integrated Copilot and direct simulation API

| Capability | Supported options |
|---|---|
| Modulation | BPSK, QPSK |
| Channel | AWGN, Rayleigh, Rician |
| Main metric | Bit Error Rate (BER) |
| SNR parameter | Eb/N0 in dB |
| Reproducibility | Configurable random seed |
| Rician configuration | Configurable K-factor |

The integrated simulation engine supports coherent detection with receiver channel knowledge for the fading-channel simulations.

Outputs include simulated BER, theoretical BER where implemented, error counts, simulation parameters, and optional received constellation samples.

### 4.2 Standalone 16-QAM implementation

The standalone QAM modules provide:

- Gray-coded 16-QAM mapping and demapping.
- Unit-average-symbol-energy constellation.
- 16-QAM BER simulation over AWGN, Rayleigh, and Rician channels.
- Theoretical BER reference for the AWGN implementation.
- Reproducible simulations using a random seed.

The standalone 16-QAM implementation is not currently integrated into the Copilot API execution path.

### 4.3 Standalone OFDM implementation

The standalone OFDM module provides:

- QPSK and 16-QAM modulation.
- IFFT-based OFDM modulation and FFT-based demodulation.
- Cyclic-prefix insertion and removal.
- AWGN and multipath Rayleigh channel simulation.
- Frequency-domain channel equalization.
- BER-versus-Eb/N0 evaluation.
- Reproducible simulations using a random seed.

The current OFDM implementation uses ideal frequency-domain channel knowledge for its Rayleigh equalization. It should not be described as a complete practical channel-estimation system.

The standalone OFDM implementation is not currently integrated into the Copilot API execution path.

---

## 5. Project Structure

The following is a logical overview of the main project areas. It is not intended to include every test, fixture, report, or local environment file.

```text
ai_wireless_copilot/
│
├── app/
│
├── backend/
│   ├── main.py
│   └── schemas.py
│
├── evaluation/
│   └── reports/
│       ├── development/
│       └── heldout/
│
├── experiments/
│
├── llm/
│   ├── client.py
│   └── validator.py
│
├── rag/
│   ├── knowledge_base/
│   │   └── wireless_basics.md
│   └── retriever.py
│
├── runner/
│
├── scripts/
│   └── evaluate_llm.py
│
├── tests/
│   ├── fixtures/
│   ├── test_api.py
│   ├── test_benchmark_dataset.py
│   ├── test_copilot.py
│   ├── test_evaluate_llm.py
│   ├── test_llm_client.py
│   ├── test_llm_validator.py
│   ├── test_rag.py
│   ├── test_wireless.py
│   ├── test_qam.py
│   ├── test_qam_simulation.py
│   └── test_ofdm.py
│
├── training/
│
├── validation/
│
├── wireless/
│   ├── simulation.py
│   ├── qam.py
│   ├── qam_simulation.py
│   └── ofdm.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

Some directories may be placeholders or contain additional files not shown in this overview. The `runner/` directory is not described as an implemented execution component.

---

## 6. Prerequisites

- Python 3.11
- Git
- Ollama installed and available from the terminal
- Sufficient system memory to run the selected local models

Default Ollama models:

- LLM: `qwen2.5:3b`
- Embedding model: `nomic-embed-text`

---

## 7. Installation and Setup

### Step 1: Clone the repository

```powershell
git clone https://github.com/pranshusharan10-hue/ai_wireless_copilot.git
cd ai_wireless_copilot
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the Python version:

```powershell
python --version
```

Use Python 3.11 for this project.

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Install and pull Ollama models

Make sure Ollama is installed and running.

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

The LLM is used for natural-language configuration extraction. The embedding model generates vectors for wireless-domain knowledge retrieval.

### Step 5: Start the API

From the repository root:

```powershell
python -m uvicorn backend.main:app --reload
```

The API is available at:

- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

The natural-language Copilot endpoint requires the Ollama service and configured models to be available.

---

## 8. API Reference

The FastAPI backend exposes the following main endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API health check |
| POST | `/api/v1/simulations/validate` | Validate a simulation request |
| POST | `/api/v1/simulations/run` | Execute a direct simulation |
| POST | `/api/v1/copilot/run` | Execute the natural-language Copilot workflow |

### 8.1 Validate a simulation

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

### 8.2 Run a direct simulation

`POST /api/v1/simulations/run`

Example request:

```json
{
  "prompt": "Simulate QPSK over Rayleigh fading",
  "modulation": "QPSK",
  "channel": "Rayleigh",
  "snr_start_db": 0,
  "snr_stop_db": 10,
  "snr_step_db": 2,
  "num_bits": 10000,
  "random_seed": 42
}
```

The integrated API execution path supports BPSK and QPSK over AWGN, Rayleigh, and Rician channels.

### 8.3 Run the natural-language Copilot

`POST /api/v1/copilot/run`

Example request:

```json
{
  "prompt": "Simulate BPSK over Rayleigh fading from 0 to 10 dB in steps of 2 dB using 10000 bits"
}
```

To include received constellation samples:

```json
{
  "prompt": "Simulate BPSK over Rayleigh fading from 0 to 10 dB in steps of 2 dB using 10000 bits",
  "include_constellation": true
}
```

Constellation samples are omitted by default to keep the response compact.

For complete request and response schemas, use the interactive documentation at `http://127.0.0.1:8000/docs`.

---

## 9. LLM Benchmark Evaluation

The project includes separate development and held-out datasets for evaluating the conversion of natural-language requests into structured simulation configurations.

### Evaluation configuration

| Setting | Value |
|---|---|
| Model | `qwen2.5:3b` |
| Runtime | Local Ollama |
| Temperature | 0 |
| Simulation execution | Disabled |
| Required configuration fields | 7 |

### Development benchmark

Previously recorded results:

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

Previously recorded results:

| Metric | Result |
|---|---:|
| Total cases | 16 |
| Exact configuration matches | 16/16 |
| Field-level accuracy | 100% |
| Validation accuracy | 100% |
| Valid case acceptance | 100% |
| Invalid case rejection | 100% |
| Average latency | 8.63 seconds |
| Median latency | 7.82 seconds |
| Maximum latency | 20.96 seconds |
| Timeouts | 0 |

These are historical benchmark results and should be checked against the actual machine-readable reports before publishing a new release.

### Run the evaluations

From the repository root:

```powershell
python -m scripts.evaluate_llm --dataset development
python -m scripts.evaluate_llm --dataset heldout
```

Reports are generated under:

```text
evaluation/reports/development/
evaluation/reports/heldout/
```

**Interpretation:** These results describe the particular development and held-out datasets used for evaluation. They do not establish 100% accuracy for arbitrary or unseen prompts. Simulation execution was disabled during these runs so that configuration extraction and validation could be evaluated independently.

---

## 10. Testing and Code Quality

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

Apply formatting if required:

```powershell
python -m ruff format .
```

Check whitespace and patch errors:

```powershell
git diff --check
```

The tests cover the integrated API, LLM configuration extraction, validation, RAG behavior, security regressions, BPSK/QPSK simulation, standalone 16-QAM behavior, and OFDM modulation and simulation.

Update the test count in this README only after running the full test suite.

---

## 11. Development Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Environment and project foundation | Complete |
| Phase 1 | BPSK/QPSK simulation over AWGN | Complete |
| Phase 2 | Rayleigh/Rician channel modeling and validation | Complete |
| Phase 3 | Ollama, RAG, validation, and Copilot API | Complete |
| Phase 4.1A | Benchmark datasets and integrity tests | Complete |
| Phase 4.1B | LLM development and held-out evaluation | Complete |
| Phase 4.2 | Error classification and handling | Complete |
| Phase 4.3 | Bounded automatic correction | Complete |
| Phase 4.4 | Safety and regression testing | Complete |
| Phase 4.5 | Documentation and release preparation | In progress |
| Phase 5 | Wireless optimization | Planned |
| Phase 6 | Optional fine-tuning and model evaluation | Planned |

### Phase 5 — Wireless Optimization

Potential future work includes adaptive modulation, parameter selection, and optimization under BER or other communication constraints.

### Phase 6 — Optional Fine-Tuning and Model Evaluation

Evaluate whether fine-tuning or parameter-efficient techniques such as QLoRA improve domain-specific configuration extraction.

These phases are planned and should not be considered implemented.

---

## 12. Reproducibility

The simulation implementations support explicit random seeds and configurable simulation parameters.

For repeatable experiments:

- Keep the random seed fixed.
- Record the modulation and channel configuration.
- Record the Eb/N0 range and bit count.
- Retain benchmark datasets and generated reports.
- Record the local Ollama model and runtime configuration.

LLM output and inference latency may vary with the runtime environment and hardware.

---

## 13. Safety, Validation, and Limitations

- Natural-language requests are converted into structured configuration values rather than executable code.
- LLM output is validated before reaching the integrated simulation engine.
- At most one bounded correction attempt is made for an invalid extracted configuration.
- Invalid configurations must not reach the simulator.
- Retrieved RAG content is reference material and must not override explicit user parameters.
- Malformed model responses and service failures are handled through controlled error paths.
- The Copilot API execution path supports BPSK and QPSK over AWGN, Rayleigh, and Rician channels.
- Standalone 16-QAM and OFDM implementations exist but are not integrated into the Copilot API execution path.
- The OFDM Rayleigh implementation uses ideal channel knowledge for equalization.
- The 16-QAM fading simulations do not claim an AWGN theoretical BER reference.
- Benchmark accuracy applies only to the included datasets.
- The Copilot depends on a running local Ollama service and the required models.
- Wireless optimization and optional fine-tuning remain planned work.

---

## 14. Phase Completion and Release Checklist

Before marking a development phase complete:

1. Confirm that the planned implementation is complete.
2. Run relevant functional and regression tests.
3. Run the full automated test suite.
4. Run Ruff lint and formatting checks.
5. Update documentation for implemented features, architecture, setup, API behavior, dependencies, and limitations.
6. Inspect the Git diff.
7. Run `git diff --check`.
8. Verify that secrets, local environment files, temporary outputs, and unrelated files are not staged.
9. Commit implementation, tests, and documentation with a descriptive commit message.
10. Push the reviewed changes to GitHub.
11. Update the roadmap only after the acceptance criteria have been met.

Do not mark planned or partially implemented capabilities as complete.
