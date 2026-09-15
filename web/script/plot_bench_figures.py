"""
plot_bench_figures.py
======================
Generates the three benchmark figures embedded on the /web `BenchPage`
(web/src/pages/bench.rs), matching the visual conventions of
`paper/script/plot_*.py` (Agg backend, serif 10pt rcParams, the same hex
color palette, dpi=150/bbox_inches="tight" save pattern).

Data: `benches/scheduler_efficiency.rs`, run via
`cargo bench --bench scheduler_efficiency`, on a quiet 4-core x86_64 dev
machine (not the CI runner — see the page's own methodology note for why
CI numbers and these numbers are expected to differ in absolute terms).
Spawn+Join and Work Deflection use the finer 8-point N sweep gathered
while investigating why dtact's advantage over Tokio is not monotonic in
N (see bench.rs's "Where the advantage comes from" section); Yield Fast
Path uses the default 3-benchmark run.

Run from repo root:
    python3 web/script/plot_bench_figures.py
Output: web/public/bench/*.png
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

OUT_DIR = Path(__file__).parent.parent / "public" / "bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Ns = np.array([1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000])

# Per-task cost in nanoseconds (median total time / N) — the metric that
# actually makes the non-monotonic dtact-vs-Tokio crossover visible; the
# raw total-time curves are both just "grows with N" and hide it. See
# bench.rs's "Where the advantage comes from" section for the reading of
# this shape (Tokio's fixed per-batch cost dominates at low N and
# amortizes away by ~N=4000; Tokio's per-task allocator pressure then
# grows and starts dominating from N=32000 on, while dtact's fixed-
# capacity pool keeps its own per-task cost roughly flat).
spawn_join_dtact_ns = np.array(
    [197.25, 341.66, 433.13, 475.96, 491.95, 501.0, 478.75, 456.5]
)
spawn_join_tokio_ns = np.array(
    [1046.8, 536.0, 493.38, 510.84, 548.0, 670.4, 634.9, 645.06]
)

deflect_dtact_ns = np.array([327.12, 469.91, 566.4, 550.1, 535.8, 517.9, 512.1, 534.4])
deflect_tokio_ns = np.array([788.71, 605.2, 545.3, 494.7, 512.8, 626.3, 686.8, 650.7])


def plot_two_panel(dtact_ns, tokio_ns, title, out_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1.plot(Ns, dtact_ns, "o-", color=BLUE, lw=2, ms=6, label="dtact")
    ax1.plot(Ns, tokio_ns, "s-", color=ORANGE, lw=2, ms=6, label="Tokio")
    ax1.set_xscale("log")
    ax1.set_xlabel("Task count $N$")
    ax1.set_ylabel("Median time per task (ns)")
    ax1.set_title(f"{title}: per-task cost")
    ax1.legend(loc="best")
    ax1.grid(True, which="both", alpha=0.3)

    ratio = tokio_ns / dtact_ns
    ax2.plot(Ns, ratio, "o-", color=GREEN, lw=2, ms=6)
    ax2.axhline(1.0, color=GREY, lw=1, linestyle="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("Task count $N$")
    ax2.set_ylabel("dtact advantage (Tokio / dtact)")
    ax2.set_title(f"{title}: advantage ratio")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.annotate(
        "dtact ahead",
        xy=(Ns[0], ratio[0]),
        xytext=(Ns[0] * 1.3, ratio[0] * 0.85),
        fontsize=8,
        color=GREEN,
    )

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


plot_two_panel(
    spawn_join_dtact_ns, spawn_join_tokio_ns, "Spawn+Join Throughput", "spawn_join.png"
)
plot_two_panel(
    deflect_dtact_ns, deflect_tokio_ns, "Work Deflection (Hot Core)", "work_deflection.png"
)

# Yield Fast Path: single point, dtact vs Tokio, self-yield with nothing
# else ready to interleave with. dtact wins this one specifically because
# it never reaches the real context-switch path here — `wait_pinned`
# resolves a self `yield_now()` on its second poll, before the adaptive
# spin budget is exhausted. See bench.rs's own note on the fast-path vs.
# real-switch (`yield_to`) distinction, and
# `benches/scheduler_efficiency.rs`'s module doc comment.
fig, ax = plt.subplots(figsize=(5, 4.2))
labels = ["dtact", "Tokio"]
values_us = [93.117, 571.69]
colors = [BLUE, ORANGE]
bars = ax.bar(labels, values_us, color=colors, width=0.5)
ax.set_ylabel(r"Median time ($\mu$s)")
ax.set_title("Yield Fast Path: 10 tasks × 100 yields each")
for bar, val in zip(bars, values_us):
    ax.annotate(
        f"{val:.0f} µs",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = OUT_DIR / "yield_fast_path.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
