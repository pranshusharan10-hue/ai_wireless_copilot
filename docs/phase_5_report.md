# AI-Wireless Copilot — Phase 5 Final Report
**Research Foundation, Repository Audit, BER Baseline, and Benchmark Preparation**
**Date:** November 2026
**Status:** Complete

---

## 1. Executive Summary

Phase 5 repositions the **AI-Wireless Copilot** project toward a scientifically rigorous research direction:

> *"Towards Trustworthy LLM-Assisted Wireless Simulation: Physics-Aware Configuration Validation and BER Verification"*

Central Research Question:
> *"Can physics-aware configuration validation and BER result verification reduce silent errors in LLM-assisted link-level simulations compared with direct prompting and RAG alone?"*

In Phase 5, all planned foundation and preparation milestones were achieved without altering the existing trusted simulation core or breaking any existing API endpoints. The project preserved all 152 pre-existing tests, added 27 new tests across benchmark integrity, statistical baselines, provider validation, and experiment result classification (reaching 179 passing tests total), audited the mathematical baseline with exact confidence intervals, curated a versioned 78-prompt benchmark dataset across 8 distinct categories, formalized 6 research metrics, and created a model-independent experiment framework.

---

## 2. Implemented Capabilities and Preserved Features

### Preserved Existing Features (Zero Regressions)
- Coherent BPSK and QPSK modulation and demodulation.
- Coherent single-carrier BER simulation over AWGN, Rayleigh, and Rician fading channels with ideal receiver CSI equalization.
- Theoretical BER calculations for AWGN, Rayleigh, and Rician channels.
- Standalone Gray-coded 16-QAM modulation, demodulation, and simulation.
- Standalone OFDM simulation with cyclic prefix over AWGN and multipath Rayleigh channels.
- FastAPI endpoints:
  - `GET /health`
  - `POST /api/v1/simulations/validate`
  - `POST /api/v1/simulations/run`
  - `POST /api/v1/copilot/run`
- Ollama LLM integration (`qwen2.5:3b`) and RAG knowledge retrieval (`nomic-embed-text`).
- Bounded single-attempt correction for invalid extracted configurations.
- Full defensive security regression suite (prompt injections, unauthorized commands, out-of-domain requests).

### New Capabilities Introduced in Phase 5
1. **Mathematical Baseline Verification Harness:**
   - `scripts/run_ber_baseline.py` executes controlled Monte Carlo simulations across BPSK/QPSK over AWGN, Rayleigh, and Rician channels.
   - Computes Clopper-Pearson 95% confidence intervals and Rule-of-Three upper bounds for zero-error observations.
   - Automatically outputs machine-readable verification data in `evaluation/baselines/ber_baseline_verification.json`.
2. **Fixed 78-Case Benchmark Dataset:**
   - 78 diverse, natural-language simulation requests spanning 8 research categories (A to H).
   - Saved canonically in `data/benchmark/phase5_benchmark.json` and mirrored in `tests/fixtures/phase5_benchmark.json`.
   - Author-curated ground truth, audited against supported simulation capabilities; pending independent review (expected action, expected config, expected physical assumptions, decision rationale, analytical reference bounds).
3. **Pydantic Benchmark Schema and Validation:**
   - `benchmark/schema.py` enforces integrity: regex IDs, action-config consistency (execute requires config; reject/clarify requires null config), category coverage, and provenance.
4. **Model-Agnostic LLM Provider Interface:**
   - `llm/provider.py` abstracts inference targets into `OllamaProvider` and `OpenAICompatibleProvider`, enabling future comparisons between local Qwen 2.5 3B and remote Mistral 7B (vLLM/Kaggle/DGX) without modifying simulator or benchmark logic.
5. **Experiment Configuration and Machine-Readable Result Schemas:**
   - `runner/experiment_config.py` specifies method presets (M1, M2, M3, M4), decoding parameters, hardware metadata, and audit tracking.
   - `runner/experiment_result.py` defines per-case output structures recording raw model output, parsed config, execution decision, validation outcome, latency breakdown, and error classifications.
6. **Literature Comparison and Positioning:**
   - Rigorous comparison against 4 supervisor-assigned papers (IPCCC 2024, AutoNetSim ICNP 2026, WirelessLLM JCIN 2024, Neurocomputing 2025).
   - Explicitly flags AutoNetSim's prior art in multi-tier verification to avoid false novelty claims, sharpening the project's unique focus on **silent physical errors in link-level digital communications**.

---

## 3. Files Created and Modified

### Created Files
| File Path | Description |
|---|---|
| `docs/repository_audit_report.md` | Comprehensive repository, git history, and capability audit |
| `docs/mathematical_baseline_report.md` | Theoretical derivations, SNR conventions, Monte Carlo limitations, and baseline results |
| `docs/related_work_comparison.md` | Literature comparison matrix, AutoNetSim overlap flag, and research gap definition |
| `docs/metrics_definition.md` | Mathematical definitions for CVR, PCR, SER, TCR, BVE, and Latency Overhead |
| `docs/experiment_design.md` | Multi-model evaluation plan, reproducibility controls, Kaggle/local commands |
| `docs/phase_5_report.md` | Master Phase 5 completion report (this file) |
| `benchmark/__init__.py` | Benchmark package initialization |
| `benchmark/schema.py` | Pydantic schema for benchmark cases and dataset |
| `data/benchmark/phase5_benchmark.json` | Versioned canonical benchmark dataset (78 cases) |
| `tests/fixtures/phase5_benchmark.json` | Test fixture benchmark dataset (78 cases) |
| `scripts/run_ber_baseline.py` | Baseline execution and Clopper-Pearson CI calculation script |
| `scripts/build_phase5_benchmark.py` | Benchmark generation, validation, and serialization script |
| `evaluation/baselines/ber_baseline_verification.json` | Machine-readable baseline simulation data |
| `llm/provider.py` | Model-agnostic LLM provider abstraction |
| `runner/experiment_config.py` | Pydantic schemas for experiment configurations and presets (M1–M4) |
| `runner/experiment_result.py` | Pydantic schemas for machine-readable run outputs and metrics |
| `tests/test_phase5_benchmark.py` | Pytest suite validating benchmark integrity and category coverage (7 tests) |
| `tests/test_experiment_config.py` | Pytest suite validating experiment presets, providers, and schemas (4 tests) |

### Modified Files
| File Path | Nature of Changes |
|---|---|
| `README.md` | Updated research direction, research question, methods M1-M4, capabilities, baseline findings, and roadmap |

---

## 4. Mathematical Baseline Findings & Issues Discovered

### Simulation vs. Analytical Consistency
Controlled Monte Carlo simulations ($N_{\text{bits}} = 100,000$, $\text{seed} = 42$) confirmed high mathematical accuracy:
- **AWGN Channel:** Simulated BER matched theoretical values within 95% Clopper-Pearson confidence intervals for all points up to 8 dB ($P_b \approx 1.91 \times 10^{-4}$, 24 errors counted).
- **Rayleigh Channel:** Simulated BER tracked the theoretical curve $P_b = \frac{1}{2}(1 - \sqrt{\gamma / (1+\gamma)})$ across -2 dB to 10 dB.
- **Rician Channel ($K=6$ dB):** Simulated BER closely matched numerical quadrature of Craig's MGF integral across all points.

### Discrepancies and Limitations Identified
1. **The Zero-Error Fallacy:** At $E_b/N_0 = 10$ dB in AWGN, theoretical BER is $3.87 \times 10^{-6}$. In $N = 100,000$ bits, only 1–2 errors occur. With the default API setting of $N = 10,000$ bits, expected errors are $\approx 0.0387$, yielding **zero errors in ~96.2% of runs**. Reporting `ber_simulated = 0.0` without confidence intervals creates an unflagged silent error.
2. **SNR Convention Ambiguity ($E_s/N_0$ vs $E_b/N_0$):** For QPSK ($k=2$), $E_s/N_0 = E_b/N_0 + 3.01$ dB. Prompts requesting $E_s/N_0$ that are directly mapped to `snr_start_db` cause an undetected 3.01 dB silent error.
3. **Physical Contradictions:** Existing schema validation permits physical contradictions (e.g., Rician $K$-factor specified with an AWGN channel; snr_start > snr_stop if validated outside Pydantic).

---

## 5. Benchmark Dataset Statistics

- **Total Cases:** 78 prompts across 8 research categories
- **Aggregate Action Breakdown:**
  - `execute`: 20 cases
  - `clarify`: 10 cases
  - `reject`: 48 cases
  - **Total:** 78 cases
- **Category Breakdown & Action Distribution:**
  - `A_standard_supported`: 10 cases (10 execute, 0 clarify, 0 reject)
  - `B_incomplete_request`: 10 cases (8 execute with standard defaults, 2 clarify, 0 reject)
  - `C_contradictory_request`: 10 cases (0 execute, 0 clarify, 10 reject)
  - `D_invalid_parameters`: 10 cases (0 execute, 0 clarify, 10 reject)
  - `E_snr_unit_confusion`: 8 cases (2 execute with converted $E_b/N_0$, 4 clarify, 2 reject)
  - `F_unsupported_mod_channel`: 10 cases (0 execute, 0 clarify, 10 reject)
  - `G_unsupported_assumptions`: 10 cases (0 execute, 0 clarify, 10 reject)
  - `H_clarify_or_reject`: 10 cases (0 execute, 4 clarify, 6 reject)
- **Review Integrity Status:**
  - All 78 benchmark prompts and expected configurations are author-curated and audited against `wireless/simulation.py` capabilities at git commit `f360d6d`.
  - In accordance with rigorous scientific review standards, their official schema status is declared as **`pending_review`** (100% of cases) pending formal independent/blinded external reviewer sign-off. Provenance strings or internal repository audits are not claimed as independent external review.

---

## 6. Verification and Regression Testing Results

```powershell
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\Prans\ai_wireless_copilot
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1
collected 179 items

tests\test_api.py .............                                          [  7%]
tests\test_benchmark_dataset.py ......                                   [ 10%]
tests\test_ber_baseline.py ....                                          [ 12%]
tests\test_copilot.py ......................                             [ 25%]
tests\test_evaluate_llm.py .....                                         [ 27%]
tests\test_experiment_config.py .............                            [ 35%]
tests\test_llm_client.py ............                                    [ 41%]
tests\test_llm_validator.py .........                                    [ 46%]
tests\test_ofdm.py ..............                                        [ 54%]
tests\test_phase5_benchmark.py ..........                                [ 60%]
tests\test_qam.py .............                                          [ 67%]
tests\test_qam_simulation.py .......................                     [ 80%]
tests\test_rag.py .............                                          [ 87%]
tests\test_wireless.py ......................                            [100%]

============================ 179 passed in 20.12s =============================
```

- **Ruff Lint Check (`ruff check .`):** Pass ("All checks passed!").
- **Ruff Format Check (`ruff format --check .`):** Pass (51 files checked and formatted).
- **Pip Check (`python -m pip check`):** Pass ("No broken requirements found.").
- **Git Diff Check (`git diff --check`):** Clean (no whitespace errors or merge markers).

---

## 7. Commands to Run Locally with Qwen 2.5 3B

```powershell
# 1. Start Ollama and verify models
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. Run baseline simulation audit
python scripts/run_ber_baseline.py

# 3. Build and test Phase 5 benchmark
python scripts/build_phase5_benchmark.py
pytest tests/test_phase5_benchmark.py tests/test_experiment_config.py

# 4. Run the full regression test suite
pytest
```

---

## 8. Next Steps for Phase 6

1. **Implement Physics-Aware Validation Layer (M3):**
   - Add checks for $E_s/N_0$ vs $E_b/N_0$ convention mismatches and auto-compensate with user notification.
   - Add physical contradiction filters (e.g., rejecting Rician $K$-factor with AWGN).
   - Add Monte Carlo sample size checks enforcing $N_{\text{bits}} \ge 100 / P_b(\text{SNR}_{\text{max}})$ to avoid zero-error starvation.
2. **Implement Independent Closed-Form BER Verifier (M4):**
   - Implement post-simulation analytical gating using Clopper-Pearson 95% confidence intervals.
   - Automatically flag zero-error observations and output the Rule-of-Three upper bound.
3. **Execute Comparative Evaluation Across Methods M1, M2, M3, M4:**
   - Run the 78 benchmark prompts locally on Qwen 2.5 3B.
   - Run the comparative benchmark on Mistral 7B (via Kaggle GPU or DGX).
   - Quantify reduction in Configuration Validity Rate (CVR), Physical Correctness Rate (PCR), and Silent-Error Rate (SER).
