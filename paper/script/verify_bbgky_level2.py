"""
verify_bbgky_level2.py
=======================
Exactly solves the H=1 load-balance CTMC (Sec. "The Exact Many-Body
Master Equation", chap:kinetic_theory) via the aggregated count chain
of Theorem thm:count-chain-exact, and computes the closed-form pair
and triplet correlators g_2, g_3 (Corollary cor:g2g3-explicit) across
N, confirming:
  - g_2 = O(1/(N-1)), matching Proposition prop:1overN.
  - g_3 = O(1/(N-1)^2), matching the order Proposition
    prop:residual-order established by a general (non-explicit)
    propagation-of-chaos argument -- here confirmed by an independent,
    fully explicit closed-form computation.
  - g_3/g_2 = O(1/(N-1)) (the multiplicative structure noted in
    Proposition prop:g3-scaling), including the load-dependent sign change
    this exact computation surfaces (g_3 > 0 at low rho_0, g_3 < 0 at
    moderate/high rho_0 -- g_2 is sign-definite by Prop anticorr, g_3
    is not).

Because rates depend on worker state only through the COUNT of
overloaded workers (H=1: each worker is binary), the full 2^N-state
chain collapses exactly (no approximation) to an (N+1)-state
birth-death chain on that count -- see thm:count-chain-exact's proof.
This is what makes an exact, closed-form g_3 tractable at all; for
H > 1 only the general order argument (Prop residual-order) is
available, not a closed form.

Run from repo root:
    python script/verify_bbgky_level2.py
Output: figure/bbgky_level2_scaling.png
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


def stationary_count_chain(N, lam0, mu):
    """Exact stationary distribution of the count chain (birth-death
    product formula), Theorem thm:count-chain-exact. Accumulates in
    log-space: the raw product telescopes to extreme magnitudes well
    before N reaches a few thousand, overflowing/underflowing a direct
    running product."""
    log_pi = np.zeros(N + 1)
    for k in range(N):
        up = (N - k) * lam0 * (1 + k / (N - 1))
        down = (k + 1) * mu
        log_pi[k + 1] = log_pi[k] + np.log(up) - np.log(down)
    log_pi -= log_pi.max()
    pi = np.exp(log_pi)
    pi /= pi.sum()
    return pi


def moments(N, lam0, mu):
    """Exact f_1, f_2, f_3, g_2, g_3 at the all-overloaded corner
    (q=q'=q''=1), Theorem thm:count-chain-exact / Corollary
    cor:g2g3-explicit."""
    pi = stationary_count_chain(N, lam0, mu)
    ks = np.arange(N + 1)
    f1 = np.sum(pi * ks / N)
    f2 = np.sum(pi * ks * (ks - 1) / (N * (N - 1)))
    f3 = np.sum(pi * ks * (ks - 1) * (ks - 2) / (N * (N - 1) * (N - 2)))
    g2 = f2 - f1**2
    g3 = f3 - f1**3 - 3 * f1 * g2
    return f1, f2, f3, g2, g3


Ns = np.array([3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 80, 120, 200, 300, 500])
rho0_values = [0.3, 0.8, 1.5]
colors = [BLUE, ORANGE, GREEN]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for rho0, color in zip(rho0_values, colors):
    g2s, g3s = [], []
    for N in Ns:
        _, _, _, g2, g3 = moments(int(N), rho0, 1.0)
        g2s.append(g2)
        g3s.append(g3)
    g2s, g3s = np.array(g2s), np.array(g3s)

    ax1.plot(Ns, np.abs(g2s), "o-", color=color, lw=1.6, ms=4,
              label=rf"$|g_2|$, $\rho_0={rho0}$")
    ax1.plot(Ns, np.abs(g3s), "s--", color=color, lw=1.2, ms=4, alpha=0.7,
              label=rf"$|g_3|$, $\rho_0={rho0}$")

    ratio = g3s / g2s
    ax2.plot(Ns, (Ns - 1) * ratio, "o-", color=color, lw=1.6, ms=4,
              label=rf"$\rho_0={rho0}$")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Worker count $N$")
ax1.set_ylabel(r"$|g_2(1,1)|$, $|g_3(1,1,1)|$")
ax1.set_title(r"Correlator magnitude: $g_2=O(1/N)$, $g_3=O(1/N^2)$")
ax1.legend(loc="best", fontsize=7)
ax1.grid(True, which="both", alpha=0.3)

ax2.set_xscale("log")
ax2.axhline(0.0, color=GREY, lw=1, linestyle="--")
ax2.set_xlabel(r"Worker count $N$")
ax2.set_ylabel(r"$(N-1)\, g_3/g_2$")
ax2.set_title(r"$g_3/g_2 = O(1/N)$, load-dependent sign")
ax2.legend(loc="best", fontsize=8)
ax2.grid(True, which="both", alpha=0.3)

fig.tight_layout()
out = OUT_DIR / "bbgky_level2_scaling.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# Print the exact-verification table referenced in the text (N=3, H=1).
print("\nExact N=3, H=1 verification (Corollary cor:g2g3-explicit):")
for rho0 in rho0_values:
    f1, f2, f3, g2, g3 = moments(3, rho0, 1.0)
    print(f"  rho0={rho0:.1f}: f1={f1:.6f} g2={g2:.6e} g3={g3:.6e}")
