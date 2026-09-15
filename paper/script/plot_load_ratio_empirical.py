"""
plot_load_ratio_empirical.py
==============================
Generates figure/load_ratio_empirical.png: mean per-task latency, DTA vs.
WS, across the offered-load sweep rho_0 in [0.1, 0.99], Empirical
Validation, Sec. "Load-Ratio Regime".

Data: one representative run of dtact's benches/dta_load_ratio.rs at
N=4 (open-loop Poisson-arrival workload; see that bench's module docs
for the full methodology). Individual points are noisy (see the
surrounding text) -- this figure illustrates the qualitative crossover,
not a precise curve.

Run from repo root:
    python script/plot_load_ratio_empirical.py
Output: figure/load_ratio_empirical.png
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

rho0 = np.array([0.1, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
dta_ns = np.array(
    [228670.6, 389419.9, 190264.4, 276690.1, 377860.0, 1214480.4, 267665.1, 398074.8]
)
ws_ns = np.array(
    [29185.1, 391683.8, 621867.8, 680397.1, 1990944.9, 980207.8, 950522.1, 2135797.1]
)

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.plot(rho0, dta_ns / 1000.0, "o-", color=BLUE, lw=2, ms=6, label="DTA")
ax.plot(rho0, ws_ns / 1000.0, "s-", color=ORANGE, lw=2, ms=6, label="WS")
ax.axvspan(0.6, 0.95, color=GREY, alpha=0.12, label=r"Claimed sweet spot ($\rho_0\in[0.6,0.95]$)")

ax.set_yscale("log")
ax.set_xlabel(r"Offered load $\rho_0$")
ax.set_ylabel(r"Mean per-task latency ($\mu$s, log scale)")
ax.set_title(r"Load-ratio sweep, $N=4$: DTA vs. WS mean latency")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent.parent / "figure" / "load_ratio_empirical.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
