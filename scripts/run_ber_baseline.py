"""Run controlled baseline BER simulations and compare with analytical references.

Verifies mathematical consistency for BPSK and QPSK over AWGN, Rayleigh,
and Rician channels across standard Eb/N0 ranges.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(ROOT_DIR / ".matplotlib_cache")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from wireless.simulation import simulate_ber, theoretical_ber  # noqa: E402

OUTPUT_DIR = ROOT_DIR / "evaluation" / "baselines"


def compute_clopper_pearson_interval(
    k_errors: int,
    n_bits: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute the exact two-sided Clopper-Pearson confidence interval.

    Handles edge cases:
      - k_errors == 0: lower is 0.0, upper is 1 - (alpha/2)**(1/n_bits).
      - k_errors == n_bits: lower is (alpha/2)**(1/n_bits), upper is 1.0.
      - 0 < k_errors < n_bits: Beta distribution quantiles.
    """
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")
    if k_errors < 0 or k_errors > n_bits:
        raise ValueError(f"k_errors must be in [0, {n_bits}], got {k_errors}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    alpha = 1.0 - confidence

    if k_errors == 0:
        lower = 0.0
        upper = float(1.0 - (alpha / 2.0) ** (1.0 / n_bits))
        return (lower, upper)

    if k_errors == n_bits:
        lower = float((alpha / 2.0) ** (1.0 / n_bits))
        upper = 1.0
        return (lower, upper)

    lower = float(stats.beta.ppf(alpha / 2.0, k_errors, n_bits - k_errors + 1))
    upper = float(stats.beta.ppf(1.0 - alpha / 2.0, k_errors + 1, n_bits - k_errors))
    return (lower, upper)


def compute_rule_of_three_upper_bound(
    n_bits: int,
    confidence: float = 0.95,
) -> float:
    """Compute one-sided Poisson/binomial upper bound when zero errors are observed.

    Upper bound: -ln(1 - confidence) / n_bits. For 95% confidence, ~2.996 / n_bits.
    """
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")
    alpha = 1.0 - confidence
    return float(-np.log(alpha) / n_bits)


def run_repeated_seed_analysis(
    configs: list[dict[str, Any]],
    num_seeds: int = 20,
    num_bits: int = 100_000,
) -> list[dict[str, Any]]:
    """Investigate statistical convergence across multiple independent seeds."""
    outlier_investigations = []

    for cfg in configs:
        mod = cfg["modulation"]
        chan = cfg["channel"]
        snr = cfg["snr_db"]
        k_db = cfg.get("rician_k_db", 6.0)

        theo = float(
            theoretical_ber(
                modulation=mod,  # type: ignore
                ebn0_db=[snr],
                channel=chan,  # type: ignore
                rician_k_db=k_db,
            )[0]
        )

        sim_rates = []
        for s in range(num_seeds):
            seed = (s + 1) * 101
            res = simulate_ber(
                modulation=mod,  # type: ignore
                ebn0_db=[snr],
                num_bits=num_bits,
                random_seed=seed,
                channel=chan,  # type: ignore
                rician_k_db=k_db,
            )
            sim_rates.append(float(res["ber_simulated"][0]))  # type: ignore

        arr = np.asarray(sim_rates, dtype=float)
        mean_sim = float(np.mean(arr))
        std_sim = float(np.std(arr, ddof=1))
        sem = float(std_sim / np.sqrt(num_seeds))
        z_score = float((mean_sim - theo) / sem) if sem > 0 else 0.0

        outlier_investigations.append(
            {
                "modulation": mod,
                "channel": chan,
                "ebn0_db": snr,
                "theoretical_ber": theo,
                "num_seeds": num_seeds,
                "num_bits_per_seed": num_bits,
                "ensemble_mean_ber": mean_sim,
                "ensemble_sem": sem,
                "ensemble_z_score": z_score,
                "statistically_consistent": abs(z_score) < 2.0,
            }
        )

    return outlier_investigations


def run_baseline() -> dict[str, Any]:
    """Execute standard parameter grid and return results with uncertainty."""
    snr_grid = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    num_bits = 100_000
    seed = 42

    configurations = [
        {"modulation": "BPSK", "channel": "AWGN", "rician_k_db": 6.0},
        {"modulation": "QPSK", "channel": "AWGN", "rician_k_db": 6.0},
        {"modulation": "BPSK", "channel": "Rayleigh", "rician_k_db": 6.0},
        {"modulation": "QPSK", "channel": "Rayleigh", "rician_k_db": 6.0},
        {"modulation": "BPSK", "channel": "Rician", "rician_k_db": 6.0},
        {"modulation": "QPSK", "channel": "Rician", "rician_k_db": 6.0},
    ]

    results = []

    for cfg in configurations:
        mod = cfg["modulation"]
        chan = cfg["channel"]
        k_db = cfg["rician_k_db"]

        sim_output = simulate_ber(
            modulation=mod,  # type: ignore
            ebn0_db=snr_grid,
            num_bits=num_bits,
            random_seed=seed,
            channel=chan,  # type: ignore
            rician_k_db=k_db,
        )

        ebn0_list = sim_output["ebn0_db"]
        ber_sim = sim_output["ber_simulated"]
        ber_theo = sim_output["ber_theoretical"]
        errors = sim_output["error_counts"]

        points = []
        for snr, b_sim, b_theo, err in zip(
            ebn0_list, ber_sim, ber_theo, errors, strict=True
        ):
            ci_low, ci_high = compute_clopper_pearson_interval(err, num_bits)
            rot_upper = (
                compute_rule_of_three_upper_bound(num_bits) if err == 0 else None
            )
            in_ci = bool(ci_low <= b_theo <= ci_high) if b_theo > 0 else True

            points.append(
                {
                    "ebn0_db": snr,
                    "errors": err,
                    "ber_simulated": b_sim,
                    "ber_theoretical": b_theo,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "rule_of_three_95_upper": rot_upper,
                    "theo_within_95_ci": in_ci,
                    "rel_error": (abs(b_sim - b_theo) / b_theo if b_theo > 0 else None),
                }
            )

        results.append(
            {
                "modulation": mod,
                "channel": chan,
                "rician_k_db": k_db if chan == "Rician" else None,
                "num_bits": num_bits,
                "random_seed": seed,
                "points": points,
            }
        )

    # Investigate points where single-seed fell outside interval
    investigation_targets = [
        {"modulation": "QPSK", "channel": "AWGN", "snr_db": 6.0},
        {"modulation": "BPSK", "channel": "Rayleigh", "snr_db": 8.0},
        {"modulation": "BPSK", "channel": "Rayleigh", "snr_db": 10.0},
        {"modulation": "QPSK", "channel": "Rayleigh", "snr_db": -2.0},
    ]
    investigation_results = run_repeated_seed_analysis(investigation_targets)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ber_baseline_verification.json"
    summary_data = {
        "metadata": {
            "description": "Controlled BER baseline simulations for Phase 5",
            "num_bits": num_bits,
            "random_seed": seed,
            "snr_grid_db": snr_grid,
            "confidence_level": 0.95,
            "interval_method": "two_sided_clopper_pearson_exact",
        },
        "results": results,
        "outlier_investigation": investigation_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    plot_baseline(summary_data, OUTPUT_DIR / "ber_baseline_verification.png")

    return summary_data


def plot_baseline(data: dict[str, Any], output_path: Path) -> Path:
    """Generate and save BER baseline curves with analytical references."""
    plt.figure(figsize=(10, 6))
    colors = {
        ("BPSK", "AWGN"): "#1f77b4",
        ("QPSK", "AWGN"): "#aec7e8",
        ("BPSK", "Rayleigh"): "#ff7f0e",
        ("QPSK", "Rayleigh"): "#ffbb78",
        ("BPSK", "Rician"): "#2ca02c",
        ("QPSK", "Rician"): "#98df8a",
    }
    markers = {
        ("BPSK", "AWGN"): "o",
        ("QPSK", "AWGN"): "s",
        ("BPSK", "Rayleigh"): "^",
        ("QPSK", "Rayleigh"): "v",
        ("BPSK", "Rician"): "D",
        ("QPSK", "Rician"): "p",
    }

    for res in data["results"]:
        key = (res["modulation"], res["channel"])
        color = colors.get(key, "#333333")
        marker = markers.get(key, "o")
        label_base = f"{res['modulation']} {res['channel']}"
        if res.get("rician_k_db"):
            label_base += f" (K={res['rician_k_db']} dB)"

        snrs = [p["ebn0_db"] for p in res["points"]]
        sim_bers = [p["ber_simulated"] for p in res["points"]]
        theo_bers = [p["ber_theoretical"] for p in res["points"]]

        plot_sim = [b if b > 0 else np.nan for b in sim_bers]

        plt.semilogy(
            snrs,
            plot_sim,
            marker=marker,
            linestyle="none",
            color=color,
            label=f"{label_base} (Sim)",
            markersize=6,
        )
        plt.semilogy(
            snrs,
            theo_bers,
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"{label_base} (Theo)",
        )

    plt.title("Phase 5 Wireless Mathematical Baseline: Simulated vs Theoretical BER")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(1e-6, 0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


if __name__ == "__main__":
    data = run_baseline()
    print(f"Generated baseline data with {len(data['results'])} configurations.")
    print("\nOutlier Multi-Seed Investigations:")
    for out in data["outlier_investigation"]:
        print(
            f"  {out['modulation']} {out['channel']} @ {out['ebn0_db']} dB: "
            f"Theo={out['theoretical_ber']:.6f}, "
            f"EnsembleMean={out['ensemble_mean_ber']:.6f}, "
            f"Z={out['ensemble_z_score']:.2f}, "
            f"Consistent={out['statistically_consistent']}"
        )
