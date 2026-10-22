"""Reference implementation for the chromatin kernel, ported from
github.com/pachterlab/FGP_2023 (Felce, Gorin, Pachter; BSD-2-Clause,
Copyright (c) 2023, Pachter Lab; the full licence text is in that repository).

Kept as close to the original as possible so that tests compare monod against
the authors' own numerics: the Ising transition matrix in their basis order,
their per-grid-point ODE solution of the vector generating function
(``pss_wrapper_n``, here with tight solver tolerances), their closed-form
moments (``calc_moments_generic``), and a plain Gillespie simulation of the
same reactions.
"""
import functools

import numpy as np
import scipy.integrate
import scipy.linalg


# --- construct_H.py ---------------------------------------------------------

def create_binaries(n):
    return [[int(ch) for ch in bin(i)[3:]] for i in range(2 ** n, 2 ** (n + 1))]


def _compare(s1, s2):
    if sum(s1) < sum(s2):
        return 1
    if sum(s1) > sum(s2):
        return -1
    for a, b in zip(s1, s2):
        if a > b:
            return 1
        if a < b:
            return -1
    return 0


def basis_sort(combos):
    return sorted(combos, key=functools.cmp_to_key(_compare), reverse=True)


def count_mismatch(string, periodic=False):
    count = sum(1 for i in range(len(string) - 1) if string[i] != string[i + 1])
    if periodic and string[-1] != string[0]:
        count += 1
    return count


def construct_H(k_ons, k_off, epsilon, periodic=False):
    n = len(k_ons)
    N = 2 ** n
    combos = basis_sort(create_binaries(n))
    H = np.zeros((N, N))
    for i in range(N):
        current = combos[i]
        num_mismatch = count_mismatch(current, periodic)
        flows_out = []
        for j in range(n):
            new = current.copy()
            new[j] = 1 - new[j]
            new_index = combos.index(new)
            flow = (k_ons[j] if current[j] == 0 else k_off) / epsilon ** num_mismatch
            H[i][new_index] = flow
            flows_out.append(flow)
        H[i][i] = -sum(flows_out)
    return H, combos


def ising_steady_state(k_ons, k_off, epsilon, periodic=False):
    n = len(k_ons)
    combos = basis_sort(create_binaries(n))
    out = np.zeros(len(combos))
    for i, string in enumerate(combos):
        prob = epsilon ** count_mismatch(string, periodic)
        for j in range(n):
            prob *= k_ons[j] if string[j] == 1 else k_off
        out[i] = prob
    return out / out.sum(), combos


# --- simulation.py: analytic joint distribution -----------------------------

def create_A(combos, bs):
    N, n = len(combos), len(bs)
    A = np.zeros((N, n))
    for i, combo in enumerate(combos):
        for j in range(n):
            if combo[j] == 1:
                A[i, j] = bs[j]
    return A


def get_g(mx):
    u = [np.exp(-2j * np.pi * np.arange(l) / l) - 1 for l in mx]
    g = np.meshgrid(*u, indexing="ij")
    return [gi.flatten() for gi in g]


def pss_wrapper_n(H, A, C, t, mx, rtol=1e-10, atol=1e-13):
    """FGP's ``pss_wrapper_n`` for a thermalised chain with no initial RNA,
    full complex FFT grid ``mx`` (as in the original), tight ODE tolerances."""
    N, n = A.shape
    g = np.asarray(get_g(mx)).T                       # (P, n)
    L, V = np.linalg.eig(C)
    Vinv = np.linalg.inv(V)
    coeff = np.asarray([np.asarray([(V @ np.diag(Vinv @ g.T[:, i]))[j] for i in range(g.shape[0])]).T
                        for j in range(n)])           # (n, n, P)
    w = scipy.linalg.null_space(H.T)
    w = (w / w.sum()).squeeze().astype(np.complex128)
    gf = np.zeros((g.shape[0], N), dtype=np.complex128)
    for i in range(g.shape[0]):
        def ufun(x):
            return np.asarray([np.exp(L * x) @ coeff[j][:, i] for j in range(n)])

        def rhs(tt, y):
            return -(H.T @ y + (A @ ufun(tt)) * y)
        res = scipy.integrate.solve_ivp(rhs, [t, 0.0], w.copy(), rtol=rtol, atol=atol, method="DOP853")
        gf[i] = res.y[:, -1]
    pss = np.zeros(list(mx) + [N])
    for j in range(N):
        pss[..., j] = np.real(scipy.fft.ifftn(gf[:, j].reshape(mx)))
    return pss


# --- moments.py ------------------------------------------------------------

def calc_moments_generic(H, combos, bs, ds, sites):
    n = len(bs)
    N = 2 ** n
    site1, site2 = sites
    S = np.array(combos)
    pi = scipy.linalg.null_space(H.T)
    pi = (pi / pi.sum()).squeeze()
    Ht = H.T
    sigma_av = pi @ S
    Pi = np.diag(pi)
    sigma_prod = S.T @ Pi @ S
    site_var_1 = sigma_prod[site1][site1] - sigma_av[site1] ** 2
    site_var_2 = sigma_prod[site2][site2] - sigma_av[site2] ** 2
    site_covar = sigma_prod[site1][site2] - sigma_av[site1] * sigma_av[site2]
    site_corr = site_covar / np.sqrt(site_var_1 * site_var_2)
    B = S * bs
    B1, B2 = np.diag(B[:, site1]), np.diag(B[:, site2])
    d1, d2 = ds[site1], ds[site2]
    M1 = np.linalg.inv(d1 * np.eye(N) - Ht)
    M2 = np.linalg.inv(d2 * np.eye(N) - Ht)
    mu_cond_1 = np.linalg.inv((d1 * np.eye(N) - Ht) @ Pi) @ B1 @ pi[:, None]
    mu_cond_2 = np.linalg.inv((d2 * np.eye(N) - Ht) @ Pi) @ B2 @ pi[:, None]
    mu1 = float(np.sum(M1 @ B1 @ pi[:, None]))
    mu2 = float(np.sum(M2 @ B2 @ pi[:, None]))
    M12 = np.linalg.inv((d1 + d2) * np.eye(N) - Ht)
    mu_12 = float(np.sum(M12 @ (B1 @ Pi @ mu_cond_2 + B2 @ Pi @ mu_cond_1)))
    covar = mu_12 - mu1 * mu2
    M11 = np.linalg.inv(2 * d1 * np.eye(N) - Ht)
    M22 = np.linalg.inv(2 * d2 * np.eye(N) - Ht)
    var1 = float(np.sum(M11 @ (2 * B1) @ Pi @ mu_cond_1)) + mu1 - mu1 ** 2
    var2 = float(np.sum(M22 @ (2 * B2) @ Pi @ mu_cond_2)) + mu2 - mu2 ** 2
    corr = covar / np.sqrt(var1 * var2)
    return dict(site_mean_1=sigma_av[site1], site_mean_2=sigma_av[site2], site_corr=site_corr,
                transcript_mean_1=mu1, transcript_mean_2=mu2, transcript_var_1=var1,
                transcript_var_2=var2, transcript_covar=covar, transcript_corr=corr)


# --- Gillespie simulation of the same reactions ------------------------------

def gillespie(H, A, ds, t_end, n_cells, rng):
    """Exact simulation of the Markov-modulated system with per-species
    degradation ``ds`` (the reactions of FGP's ``propfun_generic`` without
    delays or conversions), started from the stationary chain with no RNA.
    Returns ``(states, counts)`` at ``t_end``."""
    N, n = A.shape
    w = scipy.linalg.null_space(H.T)
    w = (w / w.sum()).squeeze()
    Hoff = H.copy()
    np.fill_diagonal(Hoff, 0.0)
    states = np.empty(n_cells, dtype=int)
    counts = np.zeros((n_cells, n), dtype=int)
    for cell in range(n_cells):
        s = rng.choice(N, p=w)
        x = np.zeros(n, dtype=int)
        t = 0.0
        while True:
            rates = np.concatenate([Hoff[s], A[s], ds * x])
            total = rates.sum()
            t += rng.exponential(1.0 / total)
            if t > t_end:
                break
            r = rng.choice(rates.size, p=rates / total)
            if r < N:
                s = r
            elif r < N + n:
                x[r - N] += 1
            else:
                x[r - N - n] -= 1
        states[cell] = s
        counts[cell] = x
    return states, counts
