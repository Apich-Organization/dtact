"""
plot_beta_ratio_empirical.py
=============================
Generates figure/beta_ratio_empirical.png: measured vs. predicted
beta_WS / beta_DTA information-acquisition-cost ratio, Empirical
Validation, Sec. "Information-Acquisition Cost".

Theory: beta_WS(N)/beta_DTA(N) = (190 + 100*log2(N)) / 80  (eq. beta_ws_num,
beta_dta_num, Sec. numa_concrete).

Measured: median-of-repeats ratio from dtact's benches/numa_information_cost.rs,
mean and range across three independent full-binary reruns at N=8/16/32
(N=64 excluded: 8x oversubscribed on the 8-logical-CPU measurement machine).

Run from repo root:
    python script/plot_beta_ratio_empirical.py
Output: figure/beta_ratio_empirical.png
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def beta_ratio_theory(n):
    return (190.0 + 100.0 * np.log2(n)) / 80.0


Ns = np.array([8, 16, 32])
theory = beta_ratio_theory(Ns)
measured_mean = np.array([2.19, 3.78, 5.15])
measured_lo = np.array([1.57, 3.57, 4.23])
measured_hi = np.array([2.70, 4.01, 5.63])

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(Ns, theory, "o-", color=BLUE, lw=2, ms=7, label=r"Predicted (theory)")
ax.errorbar(
    Ns,
    measured_mean,
    yerr=[measured_mean - measured_lo, measured_hi - measured_mean],
    fmt="s-",
    color=ORANGE,
    lw=2,
    ms=7,
    capsize=4,
    label="Measured (median, 3 reruns; range shown)",
)

ax.set_xscale("log", base=2)
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel(r"Worker count $N$")
ax.set_ylabel(r"$\beta_\mathrm{WS}/\beta_\mathrm{DTA}$")
ax.set_title("Information-acquisition cost ratio: theory vs. measurement")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent.parent / "figure" / "beta_ratio_empirical.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
