# AI-Wireless Copilot — Mathematical Baseline Report
**Phase 5: BER Baseline Audit and Statistical Verification**
**Date:** November 2026

---

## 1. Mathematical Formulation and Assumptions

The integrated simulation engine in `wireless/simulation.py` implements coherent single-carrier digital communication over AWGN, Rayleigh, and Rician flat-fading channels.

### 1.1 Bit Mapping and Modulation

1. **Binary Phase Shift Keying (BPSK):**
   - Transmitted bits: $b \in \{0, 1\}$.
   - Mapping: $s = 1 - 2b \implies 0 \mapsto +1, \; 1 \mapsto -1$.
   - Constellation: Real, 1D with alphabet $\{+1, -1\}$.
   - Symbol energy: $E_s = |s|^2 = 1$.
   - Bit energy: $E_b = E_s / 1 = 1$.
   - Detection: Coherent hard decision:
     $$\hat{b} = \begin{cases} 0, & \operatorname{Re}(\tilde{y}) \ge 0 \\ 1, & \operatorname{Re}(\tilde{y}) < 0 \end{cases}$$

2. **Quadrature Phase Shift Keying (QPSK):**
   - Transmitted bits: pairs $(b_0, b_1) \in \{0, 1\}^2$ where $k = \log_2(M) = 2$.
   - Mapping:
     $$s = \frac{(1 - 2b_0) + j(1 - 2b_1)}{\sqrt{2}}$$
   - Constellation alphabet:
     $$\left\{ \frac{1+j}{\sqrt{2}}, \; \frac{-1+j}{\sqrt{2}}, \; \frac{1-j}{\sqrt{2}}, \; \frac{-1-j}{\sqrt{2}} \right\}$$
   - Average symbol energy: $E_s = \mathbb{E}[|s|^2] = \frac{1+1}{2} = 1$.
   - Average bit energy: $E_b = E_s / k = 1 / 2$.
   - Gray coding property: Adjacent constellation symbols differ by exactly 1 bit.
   - Detection: Decoupled hard decisions on in-phase and quadrature branches:
     $$\hat{b}_0 = \begin{cases} 0, & \operatorname{Re}(\tilde{y}) \ge 0 \\ 1, & \operatorname{Re}(\tilde{y}) < 0 \end{cases}, \quad \hat{b}_1 = \begin{cases} 0, & \operatorname{Im}(\tilde{y}) \ge 0 \\ 1, & \operatorname{Im}(\tilde{y}) < 0 \end{cases}$$

---

### 1.2 SNR Conventions and AWGN Generation

- **Ratio Convention:** The simulation framework standardizes exclusively on $E_b / N_0$ expressed in decibels ($\text{SNR}_{\text{dB}}$):
  $$\gamma_b = \frac{E_b}{N_0} = 10^{\frac{\text{SNR}_{\text{dB}}}{10}}$$
- **Noise Power Spectral Density ($N_0$):**
  - Total complex AWGN variance: $\sigma_w^2 = \mathbb{E}[|w|^2] = N_0$.
  - Symmetrical real and imaginary noise components: $w = w_I + j w_Q$ with $w_I, w_Q \sim \mathcal{N}(0, \sigma^2)$ where $\sigma^2 = N_0 / 2$.
  - With $E_s = 1$ and $k$ bits per symbol ($k=1$ for BPSK, $k=2$ for QPSK):
    $$E_b = \frac{E_s}{k} = \frac{1}{k} \implies \gamma_b = \frac{1}{k N_0} \implies N_0 = \frac{1}{k \gamma_b}$$
  - Variance per real dimension:
    $$\sigma^2 = \frac{1}{2 k \gamma_b} = \frac{1}{2 \cdot \text{bits\_per\_symbol} \cdot 10^{\frac{\text{SNR}_{\text{dB}}}{10}}}$$
- **Verification in Code:**
  Line 57 of `wireless/simulation.py`:
  ```python
  noise_variance_per_dimension = 1 / (2 * bits_per_symbol * ebn0_linear)
  noise = np.sqrt(noise_variance_per_dimension) * (
      rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols))
  )
  ```
  The implementation matches the derivation with zero scaling offset.

---

### 1.3 Channel Models and Equalization

1. **AWGN Channel:**
   $$y = s + w, \quad h = 1$$
2. **Rayleigh Flat-Fading Channel:**
   $$y = h s + w, \quad h = \frac{X + j Y}{\sqrt{2}}, \quad X, Y \sim \mathcal{N}(0, 1)$$
   - Average power gain: $\mathbb{E}[|h|^2] = \frac{\mathbb{E}[X^2] + \mathbb{E}[Y^2]}{2} = \frac{1+1}{2} = 1$ (Normalized).
3. **Rician Flat-Fading Channel:**
   - Parameterized by linear $K$-factor $K_{\text{lin}} = 10^{K_{\text{dB}}/10}$.
   - Line-of-sight (LOS) specular component: amplitude $\sqrt{\frac{K_{\text{lin}}}{K_{\text{lin}}+1}}$.
   - Diffuse scattered component: $\sqrt{\frac{1}{K_{\text{lin}}+1}} \frac{X+jY}{\sqrt{2}}$.
   - Channel coefficient:
     $$h = \sqrt{\frac{K_{\text{lin}}}{K_{\text{lin}}+1}} + \sqrt{\frac{1}{K_{\text{lin}}+1}} \left( \frac{X + jY}{\sqrt{2}} \right)$$
   - Average power gain:
     $$\mathbb{E}[|h|^2] = \frac{K_{\text{lin}}}{K_{\text{lin}}+1} + \frac{1}{K_{\text{lin}}+1} \cdot 1 = 1 \quad \text{(Strictly normalized)}$$
4. **Receiver Equalization (Single-Tap Zero-Forcing / MRC):**
   - Receiver assumes ideal Channel State Information (CSI) with knowledge of $h$.
   - Equalized sample:
     $$\tilde{y} = \frac{y h^*}{\max(|h|^2, 10^{-12})} = s + \frac{w h^*}{|h|^2}$$
   - Since hard-decision detection examines only the sign of the real and imaginary parts, and $|h|^2 > 0$:
     $$\operatorname{sgn}(\operatorname{Re}(\tilde{y})) = \operatorname{sgn}(\operatorname{Re}(y h^*))$$
   - This matches optimal coherent maximal ratio combining (MRC) for a single receive antenna.

---

### 1.4 Theoretical BER Equations

1. **AWGN Channel (BPSK and QPSK):**
   $$P_b = Q\left(\sqrt{2 \gamma_b}\right) = \frac{1}{2} \operatorname{erfc}\left(\sqrt{\gamma_b}\right)$$
   - Both BPSK and Gray-coded QPSK achieve identical theoretical bit error probability as a function of $E_b / N_0$.

2. **Rayleigh Fading Channel (Coherent BPSK and QPSK):**
   $$P_b = \frac{1}{2} \left( 1 - \sqrt{\frac{\gamma_b}{1 + \gamma_b}} \right)$$
   - In the asymptotic regime ($\gamma_b \gg 1$), $P_b \approx \frac{1}{4 \gamma_b}$, exhibiting diversity order 1.

3. **Rician Fading Channel (MGF Approach):**
   Craig's integral representation for the Gaussian $Q$-function applied over the moment generating function (MGF) of the instantaneous SNR $\gamma$:
   $$P_b = \frac{1}{\pi} \int_{0}^{\pi/2} M_\gamma\left( -\frac{\gamma_b}{\sin^2\theta} \right) d\theta$$
   where the MGF of a non-central chi-square distribution with Rician parameter $K$ is:
   $$M_\gamma(s) = \frac{1 + K}{1 + K - s} \exp\left( \frac{K s}{1 + K - s} \right)$$
   Evaluated numerically using SciPy's adaptive Gauss-Kronrod quadrature (`scipy.integrate.quad`).

---

## 2. Controlled Baseline Verification Results

Controlled simulations were executed across 6 configurations using $N_{\text{bits}} = 100,000$, $\text{seed} = 42$, and $E_b/N_0 \in \{-2, 0, 2, 4, 6, 8, 10\}$ dB.

### 2.1 Numerical Comparison Table

| Mod | Channel | $E_b/N_0$ (dB) | Theoretical BER | Simulated BER | Errors | 95% Clopper-Pearson CI | In CI? |
|---|---|---|---|---|---|---|---|
| **BPSK** | AWGN | -2.0 | $1.306 \times 10^{-1}$ | $1.306 \times 10^{-1}$ | 13,061 | $[0.1285, 0.1327]$ | **Yes** |
| **BPSK** | AWGN | 0.0 | $7.865 \times 10^{-2}$ | $7.920 \times 10^{-2}$ | 7,920 | $[0.0775, 0.0809]$ | **Yes** |
| **BPSK** | AWGN | 2.0 | $3.751 \times 10^{-2}$ | $3.800 \times 10^{-2}$ | 3,800 | $[0.0368, 0.0392]$ | **Yes** |
| **BPSK** | AWGN | 4.0 | $1.250 \times 10^{-2}$ | $1.208 \times 10^{-2}$ | 1,208 | $[0.0114, 0.0128]$ | **Yes** |
| **BPSK** | AWGN | 6.0 | $2.388 \times 10^{-3}$ | $2.380 \times 10^{-3}$ | 238 | $[0.0021, 0.0027]$ | **Yes** |
| **BPSK** | AWGN | 8.0 | $1.909 \times 10^{-4}$ | $2.400 \times 10^{-4}$ | 24 | $[0.00015, 0.00036]$ | **Yes** |
| **BPSK** | AWGN | 10.0 | $3.872 \times 10^{-6}$ | $2.000 \times 10^{-5}$ | 2 | $[0.000002, 0.000072]$ | **Yes** |
| **QPSK** | AWGN | 0.0 | $7.865 \times 10^{-2}$ | $7.850 \times 10^{-2}$ | 7,850 | $[0.0768, 0.0802]$ | **Yes** |
| **QPSK** | AWGN | 4.0 | $1.250 \times 10^{-2}$ | $1.267 \times 10^{-2}$ | 1,267 | $[0.0120, 0.0134]$ | **Yes** |
| **QPSK** | AWGN | 8.0 | $1.909 \times 10^{-4}$ | $1.800 \times 10^{-4}$ | 18 | $[0.00011, 0.00028]$ | **Yes** |
| **QPSK** | AWGN | 10.0 | $3.872 \times 10^{-6}$ | $1.000 \times 10^{-5}$ | 1 | $[0.0000003, 0.000056]$ | **Yes** |
| **BPSK** | Rayleigh | 0.0 | $1.464 \times 10^{-1}$ | $1.467 \times 10^{-1}$ | 14,670 | $[0.1445, 0.1489]$ | **Yes** |
| **BPSK** | Rayleigh | 6.0 | $5.304 \times 10^{-2}$ | $5.333 \times 10^{-2}$ | 5,333 | $[0.0520, 0.0547]$ | **Yes** |
| **BPSK** | Rayleigh | 10.0 | $2.327 \times 10^{-2}$ | $2.189 \times 10^{-2}$ | 2,189 | $[0.0210, 0.0228]$ | **No** (minor fade fluctuation) |
| **BPSK** | Rician ($K=6$) | 0.0 | $1.080 \times 10^{-1}$ | $1.062 \times 10^{-1}$ | 10,621 | $[0.1043, 0.1081]$ | **Yes** |
| **BPSK** | Rician ($K=6$) | 6.0 | $2.023 \times 10^{-2}$ | $2.030 \times 10^{-2}$ | 2,030 | $[0.0194, 0.0212]$ | **Yes** |
| **BPSK** | Rician ($K=6$) | 10.0 | $4.981 \times 10^{-3}$ | $5.250 \times 10^{-3}$ | 525 | $[0.004811, 0.005718]$ | **Yes** |

### 2.2 Investigation of Baseline Outliers via Repeated Independent Seeds

In a single Monte Carlo run with $\text{seed}=42$, 4 out of 42 evaluation points fell marginally outside the exact two-sided 95% Clopper-Pearson interval. When evaluating 42 independent statistical points at significance level $\alpha = 0.05$, the expected number of false positives (points outside the 95% interval purely due to random fluctuation) is $\mathbb{E}[N_{\text{out}}] = 42 \times 0.05 = 2.1$.

To assess whether these 4 isolated discrepancies represent systematic simulation bias or standard finite-sample fluctuation, a targeted multi-seed evaluation was performed specifically for these 4 operating points. The evaluation executed $S = 20$ independent pseudo-random seeds, running $N = 100,000$ bits per seed (yielding an aggregate of $2,000,000$ bits across the 20 trials for each evaluated point):

| Mod | Channel | $E_b/N_0$ (dB) | Theoretical BER | 20-Seed Ensemble Mean | Standard Error (SEM) | Ensemble $Z$-Score | Statistically Consistent? |
|---|---|---|---|---|---|---|---|
| **QPSK** | AWGN | 6.0 | $0.002388$ | $0.002420$ | $0.000036$ | $+0.88$ | **Yes** ($|Z| < 1.0, p > 0.37$) |
| **BPSK** | Rayleigh | 8.0 | $0.035459$ | $0.035598$ | $0.000101$ | $+1.38$ | **Yes** ($|Z| < 1.4, p > 0.16$) |
| **BPSK** | Rayleigh | 10.0 | $0.023269$ | $0.023370$ | $0.000096$ | $+1.05$ | **Yes** ($|Z| < 1.1, p > 0.29$) |
| **QPSK** | Rayleigh | -2.0 | $0.189008$ | $0.189367$ | $0.000362$ | $+0.99$ | **Yes** ($|Z| < 1.0, p > 0.32$) |

> [!NOTE]
> **Scope and Statistical Qualification:**
> 1. **Sample Structure:** Each ensemble point consists of 20 independent simulation trials of $100,000$ bits each (not a single monolithic $2 \times 10^6$ bit run), allowing direct measurement of sample-to-sample variance and standard error of the mean ($\text{SEM} = \sigma / \sqrt{S}$).
> 2. **Qualified Statistical Consistency Conclusion:** Across the 4 investigated points, the empirical ensemble means agree with theoretical values within $1.4$ standard errors ($|Z| \le 1.38$, two-tailed $p > 0.16$). This indicates that the isolated deviations in $\text{seed}=42$ are consistent with standard Monte Carlo finite-sample variance rather than systematic mathematical or channel normalization errors.
> 3. **Scope Limitation:** This repeated-seed analysis covers strictly the 4 investigated outlier operating points; it does not constitute mathematical proof of convergence across all possible SNR values, channel models, or modulation orders. Finite Monte Carlo simulations always carry residual statistical uncertainty.

---

## 3. Finite Monte Carlo Statistical Limitations and Vulnerabilities

The audit exposes three critical statistical vulnerabilities in the current setup:

### 3.1 The "Rule of 100 Errors" and Default Sample Under-estimation
- **Principle:** In Monte Carlo estimation of binomial error rate $p$, the estimator $\hat{p} = k / N$ has variance $\operatorname{Var}(\hat{p}) = p(1-p)/N$. The relative standard error is:
  $$\frac{\sigma(\hat{p})}{p} = \sqrt{\frac{1-p}{N p}} \approx \frac{1}{\sqrt{k}}$$
- To obtain a relative estimation error below 10% with 95% confidence, at least $k \ge 100$ errors must be counted ($N \ge 100 / p$).
- **Current Vulnerability:**
  The default `num_bits` in `backend/schemas.py` and `llm/client.py` is $N_{\text{default}} = 10,000$.
  - At $E_b/N_0 = 10\text{ dB}$ in AWGN, $p \approx 3.87 \times 10^{-6}$.
  - Expected error count: $\mathbb{E}[k] = 10,000 \times 3.87 \times 10^{-6} \approx 0.0387$.
  - By Poisson distribution:
    $$P(k = 0) = e^{-0.0387} \approx 96.2\%$$
  - In over 96% of simulation runs at 10 dB with default settings, the simulator records **0 errors**!

### 3.2 The Zero-Error Fallacy
- When $k = 0$, the current simulator reports:
  $$\text{ber\_simulated} = 0.0$$
- This is scientifically misleading: observing 0 errors out of 10,000 bits does not prove $P_b = 0$; it merely establishes an upper confidence bound.
- By the **Rule of Three**, with 95% confidence:
  $$P_b < \frac{-\ln(0.05)}{N} \approx \frac{3}{N} = 3 \times 10^{-4}$$
- An unverified copilot accepting $N_{\text{bits}} = 10,000$ for an SNR sweep up to 20 dB will produce a string of zeros, which uninformed users or downstream LLM reasoning will misinterpret as "flawless transmission" rather than statistical starvation.

### 3.3 SNR Convention Ambiguity ($E_b/N_0$ vs $E_s/N_0$ vs $S/N$)
- For BPSK ($k=1$), $E_s = E_b$, so $E_s/N_0 = E_b/N_0$.
- For QPSK ($k=2$), $E_s = 2 E_b$, meaning:
  $$\left(\frac{E_s}{N_0}\right)_{\text{dB}} = \left(\frac{E_b}{N_0}\right)_{\text{dB}} + 10 \log_{10}(2) \approx \left(\frac{E_b}{N_0}\right)_{\text{dB}} + 3.01\text{ dB}$$
- If a user prompt specifies *"Simulate QPSK at Es/N0 = 10 dB"*, but the LLM extracts `snr_start_db = 10`, the simulator will evaluate the channel at $E_b/N_0 = 10\text{ dB}$, which corresponds to $E_s/N_0 = 13\text{ dB}$. This introduces an unflagged **3.01 dB silent error**.

---

## 4. Summary of Discrepancies and Limitations

| Issue ID | Component | Description | Impact | Minimal Correction Strategy |
|---|---|---|---|---|
| **MATH-01** | `wireless/simulation.py` | Zero error count produces raw `0.0` with no bounds. | Silent misrepresentation of high-SNR performance. | Compute Clopper-Pearson / Rule-of-Three bound in verifier. |
| **MATH-02** | `backend/schemas.py` | Default `num_bits=10000` is insufficient for $\text{SNR} > 8\text{ dB}$. | Frequent zero-error observations in standard runs. | In Phase 6, physics-aware validator flags insufficient bit count. |
| **MATH-03** | `llm/client.py` | No prompt parsing for $E_s/N_0$, $S/N$, or linear SNR. | Silent 3 dB offset for QPSK when user specifies $E_s/N_0$. | In Phase 6, parser/validator detects metric mismatch. |
| **MATH-04** | `wireless/qam_simulation.py` | Theoretical BER formula exists only for AWGN. | Rayleigh/Rician 16-QAM cannot be verified analytically. | Retain as known standalone scope limitation. |
