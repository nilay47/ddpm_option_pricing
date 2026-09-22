"""
Heston characteristic function + Carr-Madan call pricing.

The CF uses the Albrecher et al. (2007) "little trap" formulation, which is
numerically stable for long maturities and avoids the branch-cut discontinuity
of the original Heston (1993) form.

Carr-Madan: the damped call price

    c(k) = e^{alpha k} C(k),     k = log K

has Fourier transform

    psi(u) = e^{-rT} phi(u - (alpha+1)i)
             / (alpha^2 + alpha - u^2 + i(2 alpha + 1) u)

so that

    C(K) = e^{-alpha k} / pi * Re int_0^inf e^{-i u k} psi(u) du.

We evaluate this integral by Simpson quadrature at the *exact* strike rather
than on an FFT grid, which removes the grid-interpolation error of the plain
FFT implementation. `carr_madan_fft` is kept as an independent cross-check.
"""

import numpy as np

from config import Heston


def heston_cf(u, h: Heston, T: float, x0: float = None):
    """
    E[ exp(i u log S_T) ] under the Heston model with drift h.drift.

    u may be complex (needed for the Carr-Madan damping shift).
    """
    u = np.asarray(u, dtype=np.complex128)
    x0 = np.log(h.S0) if x0 is None else x0
    kappa, theta, xi, rho, v0 = h.kappa, h.theta, h.xi, h.rho, h.v0

    iu = 1j * u
    m = kappa - rho * xi * iu
    d = np.sqrt(m ** 2 + (xi ** 2) * (iu + u ** 2))

    # "little trap" branch: g2 = (m - d)/(m + d), |g2| <= 1
    g2 = (m - d) / (m + d)
    edT = np.exp(-d * T)

    C = (kappa * theta / xi ** 2) * ((m - d) * T
                                     - 2.0 * np.log((1.0 - g2 * edT) / (1.0 - g2)))
    D = (v0 / xi ** 2) * (m - d) * (1.0 - edT) / (1.0 - g2 * edT)

    return np.exp(iu * (x0 + h.drift * T) + C + D)


def _cm_integrand(u, h, T, k, alpha):
    denom = alpha ** 2 + alpha - u ** 2 + 1j * (2.0 * alpha + 1.0) * u
    psi = np.exp(-h.r * T) * heston_cf(u - (alpha + 1.0) * 1j, h, T) / denom
    return np.real(np.exp(-1j * u * k) * psi)


def carr_madan_call(strike, h: Heston, T: float,
                    alpha: float = 1.5, u_max: float = 200.0,
                    n_u: int = 8192) -> float:
    """European call price by Carr-Madan damped-integrand quadrature."""
    k = np.log(strike)
    # Simpson on [0, u_max]; n_u must be even
    n_u = n_u + (n_u % 2)
    u = np.linspace(0.0, u_max, n_u + 1)
    du = u[1] - u[0]
    f = _cm_integrand(u, h, T, k, alpha)
    w = np.ones(n_u + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    integral = (du / 3.0) * np.dot(w, f)
    return float(np.exp(-alpha * k) / np.pi * integral)


def carr_madan_calls(strikes, h: Heston, T: float, **kw) -> np.ndarray:
    return np.array([carr_madan_call(float(K), h, T, **kw) for K in np.atleast_1d(strikes)])


def carr_madan_fft(h: Heston, T: float, alpha: float = 1.5,
                   N: int = 4096, eta: float = 0.25):
    """
    Classical Carr-Madan FFT. Returns (strikes, call_prices) on the log-strike
    grid. Used only as an independent cross-check of `carr_madan_call`.
    """
    lam = 2.0 * np.pi / (N * eta)
    b = N * lam / 2.0
    u = np.arange(N) * eta
    k = -b + lam * np.arange(N)

    denom = alpha ** 2 + alpha - u ** 2 + 1j * (2.0 * alpha + 1.0) * u
    psi = np.exp(-h.r * T) * heston_cf(u - (alpha + 1.0) * 1j, h, T) / denom

    simpson = np.ones(N)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    simpson[0] = 1.0
    w = eta / 3.0 * simpson

    x = np.exp(1j * b * u) * psi * w
    y = np.fft.fft(x).real
    calls = np.exp(-alpha * k) / np.pi * y
    return np.exp(k), calls


# --------------------------------------------------------------------------

def implied_vol(price, K, S0, r, T, tol=1e-10, max_iter=200):
    """Black-Scholes implied vol by bisection (diagnostics only)."""
    from math import log, sqrt, exp, erf

    def bs(sig):
        if sig <= 0:
            return max(S0 - K * exp(-r * T), 0.0)
        d1 = (log(S0 / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
        d2 = d1 - sig * sqrt(T)
        N = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
        return S0 * N(d1) - K * exp(-r * T) * N(d2)

    lo, hi = 1e-6, 5.0
    if price <= bs(lo) or price >= bs(hi):
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bs(mid) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    from config import P_PARAMS, q_params, SIM_P
    from heston import simulate, SimConfig

    q = q_params()
    T21 = 21 / 252.0
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])

    print("CF sanity: phi(0) =", heston_cf(0.0, q, T21))
    print("  E[S_T] via -i d/du at 0 (num.):",
          np.imag(np.log(heston_cf(1e-4, q, T21))) / 1e-4, "(log-scale)")

    cm = carr_madan_calls(strikes, q, T21)
    kg, cf = carr_madan_fft(q, T21)
    cm_fft = np.interp(strikes, kg, cf)

    sim = SimConfig(n_paths=400_000, H=21, seed=999)
    pth = simulate(q, sim, world="Q")
    ST = pth.S[:, -1]
    mc = np.array([np.exp(-q.r * T21) * np.maximum(ST - K, 0).mean() for K in strikes])
    se = np.array([np.exp(-q.r * T21) * np.maximum(ST - K, 0).std(ddof=1) / np.sqrt(sim.n_paths)
                   for K in strikes])

    print(f"\n{'K':>6} {'CarrMadan':>11} {'CM-FFT':>10} {'MC':>10} {'MC se':>8} "
          f"{'(CM-MC)/se':>11} {'impvol':>8}")
    for i, K in enumerate(strikes):
        iv = implied_vol(cm[i], K, q.S0, q.r, T21)
        print(f"{K:6.1f} {cm[i]:11.5f} {cm_fft[i]:10.5f} {mc[i]:10.5f} "
              f"{se[i]:8.5f} {(cm[i]-mc[i])/se[i]:11.2f} {iv*100:7.2f}%")
