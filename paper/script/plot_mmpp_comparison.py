"""
plot_mmpp_comparison.py
=========================
Plots the stationary queue-length distribution (marginalized over
phase) for the MMPP example of Corollary cor:mmpp-numeric, against
the Poisson-matched baseline at the same mean arrival rate, making
the burstiness-driven tail effect visible directly rather than only
through the two summary numbers in the corollary's text.

Run from repo root:
    python script/plot_mmpp_comparison.py
Output: figure/mmpp_queue_distribution.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_mmpp_load_balance import solve_mmpp_qbd  # noqa: E402

BLUE = "#2E86C1"
ORANGE = "#E67E22"
GREY = "#7F8C8D"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

OUT_DIR = Path(__file__).parent.parent / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def poisson_stationary(lam, mu, H):
    rho = lam / mu
    if abs(rho - 1) < 1e-12:
        return np.full(H + 1, 1 / (H + 1))
    q = np.arange(H + 1)
    pi = (1 - rho) * rho**q / (1 - rho ** (H + 1))
    return pi


mu = 1.0
H = 10
lam_h, lam_l = 1.4, 0.2
a, b = 0.5, 0.5
q_phi = np.array([[-a, a], [b, -b]])
lam_phases = np.array([lam_h, lam_l])

pd_mmpp, pi_mmpp, lam_bar, d_flux = solve_mmpp_qbd(lam_phases, q_phi, mu, H)
pi_mmpp_marginal = pi_mmpp.sum(axis=1)
pi_poisson = poisson_stationary(lam_bar, mu, H)

qs = np.arange(H + 1)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
width = 0.38
ax.bar(qs - width / 2, pi_poisson, width, color=ORANGE, label="Poisson (mean-matched)")
ax.bar(qs + width / 2, pi_mmpp_marginal, width, color=BLUE, label="MMPP (bursty)")
ax.set_xlabel(r"Queue length $q$")
ax.set_ylabel(r"Stationary probability $\pi(q)$")
ax.set_title(
    rf"Marginal queue-length distribution ($H={H}$, mean rate $\bar\lambda_0={lam_bar:.1f}$, $\mu={mu:.0f}$)"
)
ax.set_xticks(qs)
ax.legend(loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
ax.annotate(
    f"$p_d$: {pi_poisson[-1]:.3f} (Poisson) vs {pi_mmpp_marginal[-1]:.3f} (MMPP)",
    xy=(0.98, 0.62),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=8.5,
    color=GREY,
)

fig.tight_layout()
out = OUT_DIR / "mmpp_queue_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
