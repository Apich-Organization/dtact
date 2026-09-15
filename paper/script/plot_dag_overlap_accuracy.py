"""
plot_dag_overlap_accuracy.py
==============================
Visualizes the accuracy of the leading-order eligibility-correlator
approximation (eq:delta-leading, Proposition prop:dag-overlap-correction)
against the exact hypergeometric formula (eq:delta-exact), across a
sweep of overlap ratios K^2/M and fan-ins K, making visible the
Verification and scope remark's finding: relative accuracy is not
controlled by K^2/M alone, but degrades separately with K and with
small f_{ell-1}.

Run from repo root:
    python script/plot_dag_overlap_accuracy.py
Output: figure/dag_overlap_accuracy.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_dag_overlap import approx_delta, exact_delta  # noqa: E402

BLUE = "#2E86C1"
ORANGE = "#E67E22"
GREEN = "#27AE60"

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Left panel: relative error vs K^2/M, at fixed f=0.3, for several K
# (M swept so that K^2/M spans a common range for each K).
f = 0.3
for K, color in zip([2, 4, 8], [BLUE, ORANGE, GREEN]):
    Ms = np.unique(np.round(np.geomspace(K + 1, K * K / 0.005, 40)).astype(int))
    Ms = Ms[Ms > K]
    k2m = K * K / Ms
    rel_err = []
    for M in Ms:
        ex = exact_delta(int(M), K, f)
        ap = approx_delta(int(M), K, f)
        rel_err.append(abs(ex / ap - 1))
    ax1.plot(k2m, rel_err, "-", color=color, lw=1.8, label=rf"$K_\ell={K}$")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.axhline(0.1, color="gray", lw=0.8, linestyle="--")
ax1.set_xlabel(r"$K_\ell^2/M_{\ell-1}$")
ax1.set_ylabel(r"Relative error $|\,\mathrm{exact}/\mathrm{approx} - 1\,|$")
ax1.set_title(rf"Relative error vs.\ overlap ratio ($f_{{\ell-1}}={f}$)")
ax1.legend(loc="upper left")
ax1.grid(True, which="both", alpha=0.3)

# Right panel: relative error vs f_{ell-1}, at fixed K^2/M = 0.08, for
# the same K values (varying M to hold K^2/M fixed as K changes).
k2m_fixed = 0.08
fs = np.linspace(0.05, 0.95, 40)
for K, color in zip([2, 4, 8], [BLUE, ORANGE, GREEN]):
    M = K * K / k2m_fixed
    rel_err = []
    for ff in fs:
        ex = exact_delta(int(round(M)), K, ff)
        ap = approx_delta(int(round(M)), K, ff)
        rel_err.append(abs(ex / ap - 1))
    ax2.plot(fs, rel_err, "-", color=color, lw=1.8, label=rf"$K_\ell={K}$")

ax2.set_yscale("log")
ax2.axhline(0.1, color="gray", lw=0.8, linestyle="--")
ax2.set_xlabel(r"$f_{\ell-1}$")
ax2.set_ylabel(r"Relative error $|\,\mathrm{exact}/\mathrm{approx} - 1\,|$")
ax2.set_title(rf"Relative error vs.\ $f_{{\ell-1}}$ (fixed $K_\ell^2/M_{{\ell-1}}={k2m_fixed}$)")
ax2.legend(loc="upper right")
ax2.grid(True, which="both", alpha=0.3)

fig.tight_layout()
out = OUT_DIR / "dag_overlap_accuracy.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
