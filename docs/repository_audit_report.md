# AI-Wireless Copilot — Repository Audit Report
**Phase 5: Research Foundation and Benchmark Preparation**
**Date:** November 2026
**Repository:** `https://github.com/pranshusharan10-hue/ai_wireless_copilot`
**Git Branch / Commit:** `main` @ `f360d6d` ("feat: complete wireless QAM and OFDM simulations")

---

## 1. Executive Summary

This repository audit evaluates the existing codebase of **AI-Wireless Copilot** as part of its repositioning towards the core research question:

> *"Can physics-aware configuration validation and BER result verification reduce silent errors in LLM-assisted link-level simulations compared with direct prompting and RAG alone?"*

The audit confirms that the existing implementation possesses functional simulation algorithms for BPSK and QPSK, standalone modules for 16-QAM and OFDM, a working FastAPI backend, a RAG pipeline utilizing local Ollama embeddings, and a baseline test suite of 152 automated tests.

However, prior phases focused on syntactic JSON parameter extraction and schema-range checking. Significant research gaps remain regarding physical parameter consistency, SNR convention disambiguation, finite Monte Carlo sample size adequacy, and independent BER verification.

---

## 2. Git and Repository Health

- **Working Tree Status:** Clean; no uncommitted or untracked changes at Phase 5 start.
- **Recent Git History:**
  - `f360d6d` — `feat: complete wireless QAM and OFDM simulations`
  - `42b2720` — `chore: add manual integration checks and clean repository`
  - `6da2844` — `docs: update project README for phase 4.5`
  - `858d058` — `test: strengthen copilot security regression coverage`
  - `3ee4329` — `feat: add bounded LLM response correction`
- **Dependency Environment:**
  - Python version: 3.11.9
  - Tooling: Pytest 9.1.1, Ruff 0.17.2, NumPy 1.26+, SciPy 1.13+, Matplotlib, FastAPI, Pydantic v2, HTTPX.
  - `pip check`: Pass ("No broken requirements found").
  - `ruff check .`: Pass ("All checks passed").
  - `ruff format --check .`: Pass (35 files already formatted).
  - Test Suite: 152 passed in 8.47s.

---

## 3. Detailed Component Audit

### 3.1 Wireless Simulation Core (`wireless/`)
- **`wireless/simulation.py`**:
  - Implements unit-energy BPSK ($s = 1 - 2b$) and Gray-coded unit-energy QPSK ($s = \frac{(1-2b_0) + j(1-2b_1)}{\sqrt{2}}$).
  - Channels supported: AWGN ($h=1$), Rayleigh flat fading ($h \sim \mathcal{CN}(0, 1)$), and Rician flat fading ($K$-factor parameterized).
  - Channel normalization: $\mathbb{E}[|h|^2] = 1$ for all channels.
  - Receiver equalization: Single-tap zero-forcing / maximal ratio combining with ideal receiver channel state information (CSI) ($\tilde{y} = \frac{y h^*}{|h|^2}$).
  - Theoretical BER:
    - AWGN: $P_b = \frac{1}{2}\operatorname{erfc}(\sqrt{\gamma_b})$.
    - Rayleigh: $P_b = \frac{1}{2}(1 - \sqrt{\gamma_b / (1+\gamma_b)})$.
    - Rician: Craig's MGF numerical integration via SciPy `quad`.
  - Finite Monte Carlo simulation: `simulate_ber` returns empirical bit error rate, theoretical bit error rate, integer error counts, constellation points, and metadata.
- **`wireless/qam.py` & `wireless/qam_simulation.py`**:
  - Implements 16-QAM Gray mapping with 4 bits/symbol and energy normalization factor $1/\sqrt{10}$.
  - Standalone simulations over AWGN, Rayleigh, and Rician.
  - Analytical formula implemented only for AWGN ($P_b \approx \frac{3}{4} Q(\sqrt{\frac{4}{5} \frac{E_b}{N_0}})$). Theoretical curves for Rayleigh and Rician are absent.
  - *Status:* Standalone module; not exposed via the primary API execution pipeline.
- **`wireless/ofdm.py`**:
  - Implements cyclic prefix OFDM with QPSK and 16-QAM subcarrier modulation over AWGN and multipath Rayleigh fading channels.
  - Ideal frequency-domain channel knowledge for equalization.
  - *Status:* Standalone module; not exposed via the primary API execution pipeline.

### 3.2 Backend API (`backend/`)
- **`backend/schemas.py`**:
  - `SimulationRequest`: Pydantic model enforcing `snr_start_db` (-20 to 60), `snr_stop_db` (> start, max 60), `snr_step_db` (> 0), `num_bits` (1,000 to 2,000,000), `rician_k_db` (0 to 30), `modulation` $\in \{\text{BPSK}, \text{QPSK}\}$, and `channel` $\in \{\text{AWGN}, \text{Rayleigh}, \text{Rician}\}$.
  - `CopilotRequest`: `prompt` (1 to 2,000 chars), `include_constellation` (bool).
- **`backend/main.py`**:
  - `GET /health`: Returns service status and phase indicator.
  - `POST /api/v1/simulations/validate`: Schema validation.
  - `POST /api/v1/simulations/run`: Dispatches directly to trusted simulation engine (`execute_simulation`).
  - `POST /api/v1/copilot/run`: RAG retrieval $\to$ LLM extraction $\to$ schema validation $\to$ bounded retry (1 attempt) $\to$ simulation execution $\to$ response formulation.

### 3.3 LLM Client and Validator (`llm/`)
- **`llm/client.py`**:
  - Communicates with local Ollama instance (`http://localhost:11434`) using model `qwen2.5:3b`.
  - System prompt instructs JSON extraction with hardcoded defaults.
  - Implements one bounded retry pass if validation fails, passing the validation error back to Ollama.
- **`llm/validator.py`**:
  - `validate_simulation_config`: Disallows unknown fields, enforces `SUPPORTED_MODULATIONS = {"BPSK", "QPSK"}`, validates against `SimulationRequest`.
  - *Limitation:* Does not perform physical plausibility checks or detect contradictory requirements.

### 3.4 Retrieval-Augmented Generation (`rag/`)
- **`rag/retriever.py`**:
  - Embeds document chunks from `rag/knowledge_base/wireless_basics.md` via Ollama `/api/embed` using `nomic-embed-text`.
  - Calculates cosine similarity and returns top-$k$ relevant excerpts.
- **`rag/knowledge_base/wireless_basics.md`**:
  - Covers digital modulation definitions, channel fading models, and basic BER relationships.

### 3.5 Evaluation and Benchmark Infrastructure (`evaluation/`, `scripts/`, `tests/fixtures/`)
- **Existing Benchmarks**:
  - `tests/fixtures/copilot_benchmark.json`: 10 test cases (development).
  - `tests/fixtures/copilot_heldout.json`: 16 test cases (held-out).
  - Total existing dataset: 26 cases.
  - Format: `{id, prompt, expected_status, expected_config}`.
- **Evaluation Runner (`scripts/evaluate_llm.py`)**:
  - Measures configuration field extraction accuracy.
  - Does NOT execute wireless simulations or verify BER outputs.
  - Limited metric set: extraction accuracy, field-level correctness, latency.

---

## 4. Gap Analysis: Research and Benchmark Deficiencies

| Dimension | Existing State (Phase 4.5) | Required State (Phase 5+) |
|---|---|---|
| **Benchmark Size** | 26 prompts total (10 dev, 16 heldout) | 50–100 prompts covering 8 explicit categories |
| **Category Coverage** | Valid requests and basic invalid parameters | 8 categories (A to H): standard, incomplete, contradictory, invalid, SNR confusion, unsupported modulations, unsupported physical assumptions, rejection/clarification |
| **Ground Truth** | Target schema dictionary only | Provenance, expected action (execute/clarify/reject), expected physical assumptions, decision rationale, reference BER bounds |
| **Physical Validation** | Schema & range limits only | Checks on Monte Carlo sample adequacy ($N_{\text{err}} \ge 100$), SNR range consistency, physical contradictions ($K$-factor with AWGN) |
| **Silent Error Detection** | None (undetected silent failures) | Explicit metric and verification layer for silent errors |
| **BER Verification** | Displays theoretical curve if available | Formal statistical verification (confidence intervals, error counts, deviation limits) |
| **Model Independence** | Hardcoded to Ollama `qwen2.5:3b` | Model-agnostic runner configuration supporting Ollama, Kaggle, and DGX endpoints |
| **Experimental Metrics** | JSON schema match % | 6 formal research metrics (CVR, PCR, SER, TCR, BVE, LO) |

---

## 5. Potential Compatibility and Preservation Risks

1. **Simulation Engine Integrity:** `wireless/simulation.py` must remain completely untouched in its mathematical core to preserve exact numerical compatibility with existing tests.
2. **API Contract Preservation:** All existing endpoints (`/health`, `/api/v1/simulations/validate`, `/api/v1/simulations/run`, `/api/v1/copilot/run`) must preserve their schemas and HTTP status behavior.
3. **Sandbox / Loopback Socket Access:** In sandboxed Windows environments without socket permissions, tests invoking loopback HTTP connections (`TestClient` or Ollama) encounter WinError 10013. Non-network unit tests must run cleanly sandboxed.
4. **Scope Control:** Do not introduce OFDM or 16-QAM into public API routes or claim M3/M4 as operational in Phase 5.
