# Experiment Reproducibility and Model-Independent Evaluation Design
**Phase 5: Research Foundation and Benchmark Preparation**
**Project:** AI-Wireless Copilot
**Date:** November 2026

---

## 1. Multi-Model Evaluation Strategy

To rigorously test our central research question:
> *"Can physics-aware configuration validation and BER result verification reduce silent errors in LLM-assisted link-level simulations compared with direct prompting and RAG alone?"*

the evaluation framework must support multiple foundation models across different compute environments without modifying the wireless simulation engine or benchmark ground truth.

### Target Models

1. **Primary Local Baseline: Qwen 2.5 3B (`qwen2.5:3b`)**
   - **Environment:** Local workstation / developer laptop.
   - **Runtime:** Local Ollama daemon (`http://localhost:11434`).
   - **Quantization:** Default 4-bit (`q4_k_m`).
   - **Purpose:** Fast, reproducible local evaluation, iterative testing, and development validation.
2. **Comparative Benchmark: Mistral 7B (`mistralai/Mistral-7B-Instruct-v0.3` or `mistral:7b`)**
   - **Environment:** Kaggle GPU notebook (NVIDIA T4 / P100) or high-performance GPU cluster (DGX).
   - **Runtime:** vLLM, Hugging Face TGI, or remote Ollama serving OpenAI-compatible completions.
   - **Quantization:** FP16 or AWQ / 4-bit.
   - **Purpose:** Assess whether increased model scale and reasoning capacity reduce silent physical errors in M1 and M2, and whether physics-aware validation (M3) remains necessary for larger models.

> [!IMPORTANT]
> The Mistral 7B experiment is **planned** for subsequent execution; it is NOT implemented or claimed as complete in Phase 5.

---

## 2. Experimental Controls for Fair Comparison

To ensure empirical validity, all comparative runs across models and environments must hold the following strictly constant:

1. **Identical Benchmark Prompts:** All runs execute the identical 78 benchmark prompts from `data/benchmark/phase5_benchmark.json` (version `v1.0`).
2. **Identical System Instructions:** The extraction prompt specifying schema and formatting rules must be identical across M1–M4.
3. **Identical RAG Corpus and Retrieval Settings:** For M2, M3, and M4, the retrieval corpus is locked to `rag/knowledge_base/wireless_basics.md`, using identical top-$k$ ($k=3$) and cosine similarity metric.
4. **Identical Simulation Engine:** The numerical simulation executes via the trusted Python core (`wireless/simulation.py`), with identical default parameters, seed handling, and calculation logic.
5. **Deterministic Decoding Parameters:** Greedy decoding ($\text{temperature} = 0.0$, top-$p = 1.0$) is enforced across all models to ensure determinism and eliminate sampling variance.
6. **Environment Auditing:** Every experiment run records a full hardware and software snapshot (`EnvironmentMetadata`) including Python version, platform architecture, GPU availability, Git commit hash, and execution timestamp.

---

## 3. Storage Hierarchy and Machine-Readable Artifacts

Raw model outputs, parsed configurations, intermediate validation states, simulation results, and verification decisions are recorded per run without overwriting previous results.

### Directory Structure

```text
evaluation/
└── runs/
    ├── exp_qwen3b_m1_local/
    │   ├── config.json              # ExperimentConfig snapshot
    │   ├── run_summary.json         # BenchmarkRunSummary with aggregate metrics
    │   └── cases/
    │       ├── phase5_std_01.json   # CaseEvaluationResult
    │       ├── phase5_std_02.json
    │       └── ...
    ├── exp_qwen3b_m2_local/
    ├── exp_qwen3b_m3_local/
    └── exp_qwen3b_m4_local/
```

### Raw Output Retention Rule
- Model raw text (including any chain-of-thought, malformed formatting, or syntax errors) is captured verbatim in `raw_model_response`.
- If an LLM returns invalid JSON or hallucinated parameters, it is stored as-is before any fallback or validation occurs.
- Historic experiment runs are immutable; each run generates a unique timestamped or categorized `experiment_id`.

---

## 4. Portability: Local Workstation and Kaggle GPU Compatibility

The experiment runner is designed for cross-environment portability using environment variables without hardcoded system paths:

| Configuration Parameter | Environment Variable | Local Default | Kaggle / DGX Example |
|---|---|---|---|
| Inference Provider | `LLM_PROVIDER` | `ollama` | `vllm` or `openai_compatible` |
| Base URL | `OLLAMA_BASE_URL` or `LLM_BASE_URL` | `http://localhost:11434` | `http://localhost:8000/v1` |
| Model Identifier | `OLLAMA_MODEL` or `LLM_MODEL` | `qwen2.5:3b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| API Key / Auth | `LLM_API_KEY` | *(None / empty)* | Bearer token if required |
| Results Output Dir | `EXPERIMENT_OUTPUT_DIR` | `evaluation/runs` | `/kaggle/working/runs` |

### Path Resolution
All file paths (benchmark fixtures, knowledge base, simulation engines) are dynamically resolved relative to the Git repository root using `Path(__file__).resolve().parent.parent`. No hardcoded drive letters (`C:\`) or user directories are permitted in the codebase.

---

## 5. Local Execution Commands with Qwen 2.5 3B

To run the local Qwen 2.5 3B baseline with Ollama:

### Step 1: Ensure Ollama is Running and Models are Pulled

```powershell
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### Step 2: Validate Ollama Connectivity

```powershell
curl http://localhost:11434/api/tags
```

### Step 3: Run Baseline Simulation Verification

```powershell
python scripts/run_ber_baseline.py
```

### Step 4: Validate Benchmark Dataset Integrity

```powershell
python scripts/build_phase5_benchmark.py
pytest tests/test_phase5_benchmark.py tests/test_experiment_config.py
```
