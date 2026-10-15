"""Orakel (DEMO-PLAYBOOK §12): Kernroutinen gegen einen anderen Rechenweg (explizite Iteration in x-Koordinaten, CGLS in 100 Dezimalstellen, Krylov-Definition,
Monte Carlo, Krümmung per Schleife). Jeder Test läuft in unter 10 s."""

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

import itr_model as model
from inverse_generator import SplitMix64, blurred_exact, draw_data, grid, noise_sigma, operator, phantom


def _setup(n=48, w=0.03, noise=1e-3, seed=5, crime=False):
    a = operator(n, w)
    u, sv, vt = model.decompose(a)
    b = draw_data(n, w, noise, SplitMix64(seed), crime=crime)[0]
    return a, u, sv, vt, b


@pytest.mark.parametrize("steps", [1, 5, 50])
def test_landweber_matches_the_explicit_iteration_in_x_coordinates(steps):
    a, u, sv, vt, b = _setup()
    omega = model.landweber_omega(sv)
    x = np.zeros(len(b))
    for _ in range(steps):
        x = x + omega * a.T @ (b - a @ x)
    y = model.landweber_iterate(sv, b @ u, steps, omega)[0]
    assert vt.T @ y == pytest.approx(x, rel=1e-8, abs=1e-9)


def test_cgls_plain_matches_a_conjugate_gradient_on_the_normal_equations_in_x_coordinates():
    a, u, sv, vt, b = _setup()
    x = np.zeros(48)
    r = b.copy()
    s = a.T @ r
    p = s.copy()
    gam = float(s @ s)
    ys = model.cgls_plain(sv, b @ u, 20)[:, 0]
    for k in range(1, 21):
        q = a @ p
        alpha = gam / float(q @ q)
        x = x + alpha * p
        r = r - alpha * q
        s = a.T @ r
        gn = float(s @ s)
        p = s + gn / gam * p
        gam = gn
        assert vt.T @ ys[k] == pytest.approx(x, rel=1e-6, abs=1e-8), k


@pytest.mark.parametrize("k", [1, 3, 6])
def test_cgls_exact_equals_the_least_squares_solution_over_the_krylov_space(k):
    a, u, sv, vt, b = _setup()
    cols = [a.T @ b]
    for _ in range(k - 1):
        cols.append(a.T @ (a @ cols[-1]))
    kmat = np.array(cols).T
    kmat = kmat / np.linalg.norm(kmat, axis=0)
    z = np.linalg.lstsq(a @ kmat, b, rcond=None)[0]
    x_ref = kmat @ z
    y = model.cgls_exact(sv, b @ u, kmax=k)[k, 0]
    assert vt.T @ y == pytest.approx(x_ref, rel=1e-6, abs=1e-7)
    assert float(np.linalg.norm(a @ (vt.T @ y) - b)) <= float(np.linalg.norm(a @ x_ref - b)) * (1 + 1e-9)


def _cgls_decimal(s, beta, kmax):
    getcontext().prec = 100
    n = len(s)
    s = [Decimal(float(v)) for v in s]
    r = [Decimal(float(v)) for v in beta]
    y = [Decimal(0)] * n
    sr = [s[i] * r[i] for i in range(n)]
    p = sr[:]
    gam = sum(v * v for v in sr)
    out = [np.zeros(n)]
    for _ in range(kmax):
        q = [s[i] * p[i] for i in range(n)]
        alpha = gam / sum(v * v for v in q)
        y = [y[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * q[i] for i in range(n)]
        sr = [s[i] * r[i] for i in range(n)]
        gn = sum(v * v for v in sr)
        p = [sr[i] + (gn / gam) * p[i] for i in range(n)]
        gam = gn
        out.append(np.array([float(v) for v in y]))
    return np.array(out)


@pytest.mark.parametrize("seed", [1, 2])
def test_cgls_exact_matches_cgls_in_100_decimal_digits_up_to_k_40(seed):
    a, u, sv, vt, b = _setup(seed=seed)
    beta = b @ u
    ref = _cgls_decimal(sv, beta, 40)
    ex = model.cgls_exact(sv, beta, kmax=40)[:, 0]
    assert np.abs(ex - ref).max() < 1e-9
    # die übliche Rekurrenz in doppelter Genauigkeit stimmt bei k = 20 noch, weicht bei k = 40 aber weit ab (Rundungsstagnation, Experiment 4)
    pl = model.cgls_plain(sv, beta, 40)[:, 0]
    assert np.abs(pl[20] - ref[20]).max() < 1e-6 and np.abs(pl[40] - ref[40]).max() > 0.1


def test_exact_cgls_has_a_semi_convergence_minimum_and_the_residual_never_rises():
    a, u, sv, vt, b = _setup()
    x = phantom(grid(48))
    beta = b @ u
    ys = model.cgls_exact(sv, beta, kmax=40)
    err = model.rel_errors(ys[:, 0], vt @ x, float(np.linalg.norm(x)))
    res = model.residual2(sv, beta, ys)[:, 0]
    assert np.all(np.diff(res) <= 1e-15)
    k = int(np.argmin(err[1:])) + 1
    assert 5 < k < 40 and err[40] > 2 * err[k] and err[1] > 2 * err[k]


def test_cgls_with_a_complete_krylov_space_reaches_the_direct_solution_on_a_small_grid():
    a, u, sv, vt, b = _setup(n=12, w=0.03, noise=1e-3, crime=True)
    y = model.cgls_exact(sv, b @ u, kmax=40)
    assert vt.T @ y[12, 0] == pytest.approx(np.linalg.solve(a, b), rel=1e-6, abs=1e-6)
    assert np.allclose(y[20], y[12], atol=1e-8) and np.isfinite(y).all()


def test_landweber_expected_error_matches_a_monte_carlo_for_honest_and_crime_data():
    n, w, noise, steps = 48, 0.03, 1e-3, 2000
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    sigma = noise_sigma(n, w, noise)
    xn = float(np.linalg.norm(x))
    omega = model.landweber_omega(sv)
    f = model.landweber_filter(sv, steps, omega)
    for honest in (True, False):
        rho = (u.T @ (blurred_exact(grid(n), w) - a @ x)) / sv if honest else np.zeros(n)
        b = draw_data(n, w, noise, SplitMix64(11), crime=not honest, reps=3000)
        err = model.rel_errors(model.landweber_iterate(sv, b @ u, steps, omega), vt @ x, xn)
        ref = model.expected_error(sv, vt @ x, rho, sigma, f, xn)
        assert math.sqrt(float(np.mean(err ** 2))) == pytest.approx(ref, rel=0.02)


def test_the_stop_rules_match_loop_implementations():
    n = 48
    a, u, sv, vt, b = _setup()
    sigma = noise_sigma(n, 0.03, 1e-3)
    bs = draw_data(n, 0.03, 1e-3, SplitMix64(3), crime=False, reps=3) @ u
    ys = model.cgls_exact(sv, bs, kmax=30)
    res2, sol2 = model.residual2(sv, bs, ys), model.solution2(ys)
    dp, gcv, lc = model.discrepancy_stop(res2, sigma, n), model.gcv_stop(res2, n), model.lcurve_stop(res2, sol2)

    def d1(v, t):
        """Zentrale Differenz, an den Rändern einseitig (wie np.gradient bei Schrittweite 1)."""
        if t == 0:
            return v[1] - v[0]
        if t == len(v) - 1:
            return v[-1] - v[-2]
        return (v[t + 1] - v[t - 1]) / 2.0

    for j in range(3):
        first = next((k for k in range(1, 31) if res2[k, j] <= sigma ** 2 * n), 30)
        assert dp[j] == first
        g = [res2[k, j] / max(n - k, 1e-9) ** 2 for k in range(1, 31)]
        assert gcv[j] == 1 + int(np.argmin(g))
        rho = [0.5 * math.log(res2[k, j]) for k in range(1, 31)]
        eta = [0.5 * math.log(sol2[k, j]) for k in range(1, 31)]
        r1 = [d1(rho, t) for t in range(len(rho))]
        e1 = [d1(eta, t) for t in range(len(eta))]
        best, best_k = -1e300, -1
        for t in range(2, 29):                       # innere Punkte der Krümmung (np.gradient-Ränder der zweiten Ableitung ausgespart)
            r2, e2 = d1(r1, t), d1(e1, t)
            kap = (r2 * e1[t] - r1[t] * e2) / max((r1[t] ** 2 + e1[t] ** 2) ** 1.5, 1e-300)
            if kap > best:
                best, best_k = kap, t + 1            # Zeilenindex t entspricht k = t + 1
        assert lc[j] == best_k


def test_the_realization_oracle_never_loses_to_a_rule_and_the_rules_differ():
    n, w, noise = 48, 0.03, 1e-3
    x = phantom(grid(n))
    u, sv, vt = model.decompose(operator(n, w))
    sigma = noise_sigma(n, w, noise)
    b = draw_data(n, w, noise, SplitMix64(21), crime=False, reps=60) @ u
    ys = model.cgls_exact(sv, b, kmax=40)
    err = model.rel_errors(ys, vt @ x, float(np.linalg.norm(x)))
    res2, sol2 = model.residual2(sv, b, ys), model.solution2(ys)
    picks = {"dp": model.discrepancy_stop(res2, sigma, n), "gcv": model.gcv_stop(res2, n), "lcurve": model.lcurve_stop(res2, sol2), "oracle": model.oracle_stop(err)}
    rows = np.arange(60)
    for key in ("dp", "gcv", "lcurve"):
        assert np.all(err[picks["oracle"], rows] <= err[picks[key], rows] + 1e-15)
    assert float(np.median(picks["lcurve"])) > float(np.median(picks["dp"]))
