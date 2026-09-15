"""
plot_cramer_makespan.py
========================
Compares the exact makespan tail P(C_max > c) (Proposition prop:max-cdf,
chap:makespan) against the new Cramer-type exponential bound
(Theorem thm:cramer-makespan) and the existing Chebyshev-based bound
(Theorem thm:stat-makespan), at fixed benchmark-scale parameters,
showing the Cramer bound tracks the exact exponential decay rate
while Chebyshev decays only polynomially and is orders of magnitude
looser in the tail.

Run from repo root:
    python script/plot_cramer_makespan.py
Output: figure/cramer_makespan_tail.png
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2E86C1"
ORANGE = "#E67E22"
GREEN = "#27AE60"
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

# Small illustrative scale (benchmark H=114,688 makes the exponential decay
# invisible on any readable axis range); the qualitative slope comparison
# is scale-invariant in m = c/delta.
N = 4
H = 60
rho = 0.7
delta = 1.0


def F1(m):
    return (1 - rho ** (m + 1)) / (1 - rho ** (H + 1))


ms = np.arange(0, H + 1)
exact = 1 - F1(ms) ** N

# Cramer bound (thm:cramer-makespan):
# P(M_N > m) <= N/(1-rho^{H+1}) * rho^(m+1), valid for every real
# threshold (the proof floors to the nearest integer internally).
cramer = np.minimum(1.0, (N / (1 - rho ** (H + 1))) * rho ** (ms + 1))

# Chebyshev bound (thm:stat-makespan): P(Cmax > c) <= N/z^2 with
# t = z*sigma_q = (c - qbar); invert for a direct P(M_N>m) style curve
# using the same sigma_q the theorem uses (Delta l <= 1/2 => sigma_q<=H/2).
qbar = rho / (1 - rho) - (H + 1) * rho ** (H + 1) / (1 - rho ** (H + 1))
sigma_q = H / 2.0  # worst-case Delta l = 1/2 bound used by thm:stat-makespan
t = np.maximum(ms - qbar, 1e-9)
chebyshev = np.minimum(1.0, N * sigma_q**2 / t**2)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.semilogy(ms, np.maximum(exact, 1e-300), color=GREY, lw=2.5, label="Exact (Prop. max-cdf)")
ax.semilogy(ms, cramer, "--", color=BLUE, lw=2, label="Cramer bound (Thm. cramer-makespan)")
ax.semilogy(ms, chebyshev, ":", color=ORANGE, lw=2, label="Chebyshev bound (Thm. stat-makespan)")
ax.set_xlabel(r"Threshold $m = c/\delta$")
ax.set_ylabel(r"$\mathbb{P}(M_N > m)$")
ax.set_title(rf"Makespan tail bounds ($N={N}$, $H={H}$, $\rho^*={rho}$)")
ax.set_ylim(1e-16, 2)
ax.legend(loc="upper right")
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
out = OUT_DIR / "cramer_makespan_tail.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
