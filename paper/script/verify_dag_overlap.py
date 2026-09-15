"""
verify_dag_overlap.py
======================
Verifies the leading-order finite-M_{ell-1} eligibility correlator
delta(t) (Proposition prop:dag-overlap-correction, sec:dag_kinetic)
against the exact hypergeometric formula it approximates
(Lemma lem:overlap-hypergeom).

Two level-ell tasks v, v' each draw K_ell predecessors uniformly
without replacement from the M_{ell-1} level-(ell-1) tasks
(thm:dag_ode's random-DAG ensemble). The overlap size
m = |P_v cap P_v'| is exactly Hypergeometric(M_{ell-1}, K_ell,
K_ell)-distributed. This script computes the exact eligibility
correlator delta(t) = E_m[f^{2K-m}] - f^{2K} via the hypergeometric
PMF and compares it against the closed-form leading-order
approximation delta(t) ~ (K^2/M) f^{2K-1}(1-f), confirming the ratio
-> 1 as K^2/M_{ell-1} -> 0 (the regime the approximation is derived
for) and remains the right order of magnitude outside it.

Run from repo root:
    python script/verify_dag_overlap.py
"""

import numpy as np
from scipy.stats import hypergeom


def exact_delta(M, K, f):
    rv = hypergeom(M, K, K)
    ms = np.arange(0, K + 1)
    pmf = rv.pmf(ms)
    e_f_inv_m = np.sum(pmf * f ** (-ms))
    return f ** (2 * K) * (e_f_inv_m - 1)


def approx_delta(M, K, f):
    return (K**2 / M) * f ** (2 * K - 1) * (1 - f)


if __name__ == "__main__":
    print(f"{'M':>6} {'K':>4} {'K^2/M':>8} {'f':>6} {'exact':>14} {'approx':>14} {'ratio':>8}")
    for M in [50, 200, 1000]:
        for K in [2, 4, 8]:
            if K > M:
                continue
            for f in [0.3, 0.6, 0.9]:
                ex = exact_delta(M, K, f)
                ap = approx_delta(M, K, f)
                print(
                    f"{M:>6} {K:>4} {K*K/M:>8.3f} {f:>6.2f} "
                    f"{ex:>14.6e} {ap:>14.6e} {ex/ap:>8.4f}"
                )
