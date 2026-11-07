# Related Work Comparison and Research Gap Analysis
**Project:** AI-Wireless Copilot
**Phase 5:** Research Foundation and Literature Grounding
**Date:** November 2026

---

## 1. Research Context

The integration of Large Language Models (LLMs) into scientific simulation and wireless engineering has rapidly accelerated. Recent literature explores positioning LLMs as natural-language interfaces, intent compilers, and autonomous experimentation agents.

However, existing approaches primarily focus on code generation and macro-level execution success (whether the simulator executes without runtime crashes). They largely overlook **silent physical and statistical errors** in link-level digital communications.

This document systematically examines four foundational papers assigned by the project supervision, analyzes their technical overlaps, flags prior art, and articulates our specific research gap.

---

## 2. Detailed Review of Foundational Literature

### Paper 1: Revolutionizing Wireless Modeling and Simulation with Network-Oriented LLMs
- **Citation:** J. Liu, Z. Peng, D. Xu, and Y. Liu, *"Revolutionizing Wireless Modeling and Simulation with Network-Oriented LLMs,"* in *Proc. 43rd IEEE International Performance, Computing, and Communications Conference (IPCCC)*, Nov. 2024 (Best Paper Runner-up). [DOI: 10.1109/IPCCC62551.2024.10850042](https://ieeexplore.ieee.org/document/10850042/)
- **Problem Addressed:** High operational barriers and manual scripting overhead required to configure complex physical and network simulators (e.g., Sionna and ns-3) from user intent.
- **Architecture and Techniques:** Parameter-Efficient Fine-Tuning (PEFT, specifically LoRA) combined with Retrieval-Augmented Generation (RAG) to adapt general LLMs into domain-specific network assistants.
- **Simulator / Tools Used:** NVIDIA Sionna (GPU-accelerated Python library for link-level PHY simulations) and ns-3.
- **Evaluation Methodology:** Script generation accuracy, task completion speed, human-in-the-loop operational efficiency, and code executability.
- **Verification Mechanisms:** Relies on Python syntax parsing and simulator runtime execution success. Lacks closed-loop analytical result verification or physical domain consistency bounds.
- **Overlap with Our Project:** Utilizes natural language + RAG to interface with wireless simulation engines.
- **Research Gap:** Assumes that if the generated simulation script runs and returns values, the result is scientifically valid. Does not guard against silent physical parameter misunderstandings or finite-sample Monte Carlo distortion.

---

### Paper 2: AutoNetSim: Intent-Driven Wireless Network Experimentation with Self-Evolving Agents
- **Citation:** Pervasive Intelligence Lab, *"AutoNetSim: Intent-Driven Wireless Network Experimentation with Self-Evolving Agents,"* IEEE ICNP 2026 / open-source repository: [github.com/Pervasive-Intelligence-Lab/agentic-sionna](https://github.com/Pervasive-Intelligence-Lab/agentic-sionna).
- **Problem Addressed:** End-to-end automated generation of 3D radio environments, ray tracing, and system-level wireless experimentation from high-level natural language intents (e.g., *"build a 6x5 office and compute coverage at 5 GHz"*).
- **Architecture and Techniques:** Self-evolving multi-agent architecture incorporating procedural prompt routing, Mitsuba 3D scene generation, ITU material assignment, memory mechanisms, and an iterative debugging feedback loop.
- **Simulator / Tools Used:** NVIDIA Sionna 2.0 (Mitsuba ray tracing and radio propagation maps) and Flask/Three.js interactive UI.
- **Evaluation Methodology:** Pass rates on complex 3D environment synthesis, ray-tracing coverage generation, and multi-turn error correction benchmarks.
- **Verification Mechanisms:** Implements a three-tier verification harness:
  1. *Artifact checks:* Validates output file formats and scene descriptions.
  2. *Executable checks:* Verifies syntax and run-time termination.
  3. *Oracle checks:* Compares outputs against pre-defined heuristics and geometric constraints.
- **CRITICAL OVERLAP FLAG:**
  > [!IMPORTANT]
  > **AutoNetSim explicitly incorporates multi-tier simulation verification (including oracle checks). Therefore, generic LLM simulation verification CANNOT be claimed as novel by this project.**
- **Research Gap and Differentiation:**
  AutoNetSim addresses macroscopic 3D scene synthesis and ray tracing in Sionna. Our research specifically isolates **link-level physical layer digital communications** ($E_b/N_0$ vs $E_s/N_0$ convention shifts, Monte Carlo sample starvation, unnormalized fading channel power, zero-error interpretation, and closed-form analytical BER boundaries), where an LLM may generate a perfectly executable script that fails silently on scientific grounds.

---

### Paper 3: WirelessLLM: Empowering Large Language Models Towards Wireless Intelligence
- **Citation:** J. Shao, J. Tong, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang, *"WirelessLLM: Empowering Large Language Models Towards Wireless Intelligence,"* *Journal of Communications and Information Networks (JCIN)*, vol. 9, no. 2, pp. 115–130, 2024. [DOI: 10.23919/JCIN.2024.10582827](https://doi.org/10.23919/JCIN.2024.10582827)
- **Problem Addressed:** Bridging general-purpose LLM limitations when faced with rigorous telecommunication standards, multimodal network telemetry, and mathematical optimization problems.
- **Architecture and Techniques:** Proposes three pillars: Knowledge Alignment (adapting to telecom standards), Knowledge Fusion (multimodal telemetry and graph data), and Knowledge Evolution (continuous learning via environmental feedback). Employs RAG over 3GPP specifications and tool invocation.
- **Simulator / Tools Used:** External numerical solvers, convex optimizers, and standardized mathematical calculation tools.
- **Evaluation Methodology:** Qualitative and task-specific case studies: spectrum sensing in cognitive radio, power allocation optimization, and 3GPP signaling protocol troubleshooting.
- **Verification Mechanisms:** Relies on RAG document grounding to suppress hallucinations and external optimizer convergence. No statistical link-level simulation verifier is proposed.
- **Overlap with Our Project:** Utilization of RAG for wireless knowledge and structured tool invocation.
- **Research Gap:** WirelessLLM operates at the conceptual and high-level architectural tier. It does not measure or quantify silent error rates during simulation configuration extraction or evaluate statistical convergence in stochastic Monte Carlo runs.

---

### Paper 4: Creation, Evaluation and Self-Validation of Simulation Models with Large Language Models
- **Citation:** T. Möltner, P. Manzl, M. Pieber, and J. Gerstmayr, *"Creation, evaluation and self-validation of simulation models with large language models,"* *Neurocomputing*, vol. 663, p. 132030, 2026. [DOI: 10.1016/j.neucom.2025.102702](https://doi.org/10.1016/j.neucom.2025.102702)
- **Problem Addressed:** Automating the lifecycle of engineering simulation models (creation, evaluation, and validation) using LLMs, specifically examining whether LLMs can detect flaws in simulation configurations.
- **Architecture and Techniques:** Automated benchmarking harness with parameterized test cases where analytical or high-fidelity ground truth is known. LLM agents act as model creators and self-validators.
- **Simulator / Tools Used:** Python-based multibody dynamics and mechanical engineering simulation packages.
- **Evaluation Methodology:** Formulates two formal metrics: **Executability ($\eta$)** (code runs without exception) and **Correctness ($\kappa$)** (model reproduces correct physical behavior). Evaluates self-validation accuracy using Precision, Recall, and F1-score.
- **Verification Mechanisms:** Employs *LLM self-validation*, where the LLM is prompted to inspect its own code or third-party simulation code to identify incorrect assumptions or parameters.
- **Overlap with Our Project:** Explicitly separates execution validity from physical correctness; investigates automated validation of simulation models.
- **Research Gap:**
  1. *Domain limitation:* Evaluated in mechanical/multibody systems, not communication theory or stochastic wireless channels.
  2. *Mechanism limitation:* Relies on *LLM self-validation*, which suffers from LLM blind spots, sycophancy, and shared hallucination modes. Our research replaces LLM self-validation with **independent, deterministic physics-aware validation and analytical BER verification** (M3 and M4).

---

## 3. Comparative Matrix

| Feature | IPCCC 2024 (Liu et al.) | AutoNetSim (ICNP 2026) | WirelessLLM (JCIN 2024) | Neurocomputing 2025 (Möltner et al.) | AI-Wireless Copilot (Our Project) |
|---|---|---|---|---|---|
| **Domain** | Wireless PHY & Network | 3D Ray-Tracing & Propagation | Telecom Standards & Optimization | Multibody Mechanical Systems | Link-Level Digital Wireless PHY |
| **LLM Role** | Intent-to-script translator | Autonomous agent & scene builder | Telecom intelligent assistant | Model generator & self-validator | Parameter extractor & configuration generator |
| **RAG Grounding** | Domain manuals | Scene/material templates | 3GPP specs & papers | Model templates | Wireless basics knowledge base |
| **Verification Strategy** | Syntax & execution only | Artifact, executable, & oracle checks | RAG grounding & external solver | LLM self-validation | Deterministic physics rules + closed-form BER bounds |
| **Focus on Silent Errors** | No | Secondary (ray tracing anomalies) | No | Yes (mechanical modeling) | **Primary (link-level communication physics)** |
| **Simulator Execution** | Unchecked script generation | Arbitrary script generation | External optimizer execution | Script generation | Trusted parameterized execution sandbox |
| **Metric Rigor** | Execution success | Benchmark pass rate | Task accuracy | Executability $\eta$, Correctness $\kappa$ | CVR, PCR, SER, TCR, BVE, LO |

---

## 4. Articulation of the Project's Specific Research Contribution

Our research does **not** claim novelty for:
- Asking an LLM to configure a wireless simulator (explored by Liu et al. and AutoNetSim).
- Using RAG to ground communication parameters (explored by WirelessLLM and Liu et al.).
- Generic simulation verification using oracles or LLM self-checks (demonstrated by AutoNetSim and Möltner et al.).

Instead, the novelty and technical contribution of **AI-Wireless Copilot** lies in:
1. **Quantifying Silent Errors in Link-Level Simulations:** Defining, benchmarking, and measuring silent failures where an LLM configuration executes without crashing but violates physical laws, misinterprets SNR conventions ($E_b/N_0$ vs $E_s/N_0$), or starves Monte Carlo sample size ($N_{\text{err}} \to 0$).
2. **Deterministic Physics-Aware Validation vs. LLM Self-Validation:** Demonstrating that deterministic domain rules outperform LLM self-validation in catching contradictory parameters and sample-size inadequacies before execution.
3. **Closed-Form BER Verification Bounds:** Establishing automated post-simulation verification using analytical digital communication bounds (AWGN error function, Rayleigh closed form, Rician MGF Craig integral, and Clopper-Pearson / Rule-of-Three confidence intervals) to certify simulation validity independently of the generative model.
