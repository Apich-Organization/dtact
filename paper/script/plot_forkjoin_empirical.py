"""
plot_forkjoin_empirical.py
============================
Generates figure/forkjoin_empirical.png: measured fork-join makespan vs.
Theorem fj_first's first-order bound and the combined bound (first-order
+ Gumbel-variance + batch-arrival/NESS-mismatch corrections), Empirical
Validation, Sec. "Fork-Join Makespan Bound".

Data: dtact's benches/dta_forkjoin_bound.rs, the deepest/widest
configuration tested (d=8 levels, K=16-way join -- the one configuration
where deflection meaningfully engages; see that bench's module docs),
at N in {4, 6, 8}, 400-trial mean per point.

Run from repo root:
    python script/plot_forkjoin_empirical.py
Output: figure/forkjoin_empirical.png
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

Ns = np.array([4, 6, 8])
first_order_bound = np.array([1180.92, 967.58, 860.92])
combined_bound = np.array([9300.26, 9086.93, 8980.26])
dta_measured = np.array([4644.15, 7833.73, 9332.50])
ws_measured = np.array([1869.58, 2855.86, 3155.72])

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.plot(Ns, first_order_bound, "d--", color=GREY, lw=1.5, ms=7, label="First-order bound")
ax.plot(Ns, combined_bound, "d-", color=GREY, lw=2, ms=7, label="Combined bound")
ax.plot(Ns, dta_measured, "o-", color=BLUE, lw=2, ms=8, label="DTA (measured)")
ax.plot(Ns, ws_measured, "s-", color=ORANGE, lw=2, ms=8, label="WS (measured)")

ax.set_xticks(Ns)
ax.set_xlabel(r"Worker count $N$")
ax.set_ylabel(r"Makespan ($\mu$s)")
ax.set_title(r"Fork-join makespan, $d{=}8$, $K{=}16$: bounds vs. measurement")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
ax.annotate(
    "DTA crosses combined\nbound at N=8 (this\nmachine's core count)",
    xy=(8, 9332.50),
    xytext=(5.6, 6200),
    fontsize=7.5,
    color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1),
)

fig.tight_layout()
out = Path(__file__).parent.parent / "figure" / "forkjoin_empirical.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
