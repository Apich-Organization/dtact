"""
verify_mmpp_load_balance.py
=============================
Solves the finite MMPP/M/1/H self-consistency system
(Theorem thm:mmpp-sc, sec:non-poisson) for a concrete 2-phase
ON/OFF MMPP arrival process, and compares against a Poisson baseline
with the same mean arrival rate, isolating the effect of arrival
burstiness (beyond the mean rate alone) on the deflection probability
p_d and mean queue length.

The self-consistent quantity is D = sum_phi lam0(phi)*pi(H,phi)
(eq:mmpp-flux), the phase-*weighted* flux of peers actually at the
threshold -- NOT lam_bar*p_d (mean rate times phase-marginal
deflection probability), which would assume phase and queue occupancy
are independent under the stationary law. They are not: a peer's
phase modulates its own arrival rate and hence its own probability of
sitting at H, so factoring the expectation that way is wrong (see
Proposition prop:deflect-averages's proof). D is found by the same
outer fixed-point iteration as p_d was in the scalar (Poisson) case.

The joint (queue length, phase) process is a finite quasi-birth-death
(QBD) chain -- see Neuts' matrix-geometric theory for the general
(infinite-buffer) case; here H is finite, so the stationary
distribution is solved directly as the null vector of the
(H+1)*r-dimensional block-tridiagonal generator, no R-matrix fixed
point needed.

Run from repo root:
    python script/verify_mmpp_load_balance.py
"""

import numpy as np


def solve_scalar_sc(lam0, mu, H, tol=1e-12, maxit=200):
    """Standard scalar self-consistency (eq:sc): rho = lam0*delta*(1+pi(H;rho))."""
    rho = lam0 / mu
    for _ in range(maxit):
        pi_h = (1 - rho) * rho**H / (1 - rho ** (H + 1)) if abs(rho - 1) > 1e-12 else 1 / (H + 1)
        rho_new = (lam0 / mu) * (1 + pi_h)
        if abs(rho_new - rho) < tol:
            rho = rho_new
            break
        rho = rho_new
    pi_h = (1 - rho) * rho**H / (1 - rho ** (H + 1)) if abs(rho - 1) > 1e-12 else 1 / (H + 1)
    return rho, pi_h


def solve_mmpp_qbd(lam_phases, q_phi, mu, H, tol=1e-12, maxit=500):
    """Finite MMPP/M/1/H self-consistency (Theorem thm:mmpp-sc), fixed
    point in D = sum_phi lam0(phi)*pi(H,phi) (eq:mmpp-flux/eq:mmpp-sc)."""
    r = len(lam_phases)
    a_mat = np.vstack([q_phi.T, np.ones(r)])
    b_vec = np.zeros(r + 1)
    b_vec[-1] = 1
    pi_phi, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    lam_bar = pi_phi @ lam_phases

    def idx(q, p):
        return q * r + p

    d_flux = 0.0
    pi = None
    for _ in range(maxit):
        lam_eff = lam_phases + d_flux
        n = (H + 1) * r
        gen = np.zeros((n, n))
        for q in range(H + 1):
            for p in range(r):
                i = idx(q, p)
                if q < H:
                    j = idx(q + 1, p)
                    gen[i, j] += lam_eff[p]
                    gen[i, i] -= lam_eff[p]
                if q > 0:
                    j = idx(q - 1, p)
                    gen[i, j] += mu
                    gen[i, i] -= mu
                for p2 in range(r):
                    if p2 != p:
                        rate = q_phi[p, p2]
                        j = idx(q, p2)
                        gen[i, j] += rate
                        gen[i, i] -= rate
        a2 = np.vstack([gen.T, np.ones(n)])
        b2 = np.zeros(n + 1)
        b2[-1] = 1
        pi, *_ = np.linalg.lstsq(a2, b2, rcond=None)
        pi = pi.reshape(H + 1, r)
        d_new = lam_phases @ pi[H, :]
        if abs(d_new - d_flux) < tol:
            d_flux = d_new
            break
        d_flux = d_new
    pd = pi[H, :].sum()
    return pd, pi, lam_bar, d_flux


if __name__ == "__main__":
    mu = 1.0
    H = 10
    lam_h, lam_l = 1.4, 0.2  # on/off direct-arrival rates
    a, b = 0.5, 0.5  # symmetric phase-switch rates -> pi_phi = (0.5, 0.5)
    q_phi = np.array([[-a, a], [b, -b]])
    lam_phases = np.array([lam_h, lam_l])

    pd_mmpp, pi_mmpp, lam_bar, d_flux = solve_mmpp_qbd(lam_phases, q_phi, mu, H)
    rho_poisson, pd_poisson = solve_scalar_sc(lam_bar, mu, H)

    qbar_mmpp = sum(q * pi_mmpp[q, :].sum() for q in range(H + 1))
    rho = rho_poisson
    qbar_poisson = rho / (1 - rho) - (H + 1) * rho ** (H + 1) / (1 - rho ** (H + 1))

    print(f"mean rate lam_bar = {lam_bar:.4f} (matched to Poisson baseline)")
    print(f"Poisson-matched:  rho*={rho_poisson:.6f}  p_d={pd_poisson:.6f}  qbar={qbar_poisson:.4f}")
    print(f"MMPP (bursty):    D={d_flux:.6f}  p_d={pd_mmpp:.6f}  qbar={qbar_mmpp:.4f}")
    print(f"Ratio p_d^MMPP / p_d^Poisson = {pd_mmpp / pd_poisson:.4f}")
    print(f"Ratio qbar^MMPP / qbar^Poisson = {qbar_mmpp / qbar_poisson:.4f}")
