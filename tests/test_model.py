"""Kernroutinen an der 3 x 3-Mini-Instanz von Hand (DEMO-PLAYBOOK §1b); jede Einheit einzeln."""

import numpy as np
import pytest

import itr_model as m


def _coords(mini):
    a, x, b = mini
    u, sv, vt = m.decompose(a)
    return a, x, b, u, sv, vt, b @ u, vt @ x


def test_decompose_reconstructs_the_matrix_with_the_singular_values_one_half_tenth(mini):
    a = mini[0]
    u, sv, vt = m.decompose(a)
    assert np.allclose(u @ np.diag(sv) @ vt, a) and sv == pytest.approx([1.0, 0.5, 0.1])


def test_landweber_omega_is_the_factor_over_sigma1_squared(mini):
    sv = m.decompose(mini[0])[1]
    assert m.landweber_omega(sv) == pytest.approx(1.0) and m.landweber_omega(sv, 1.5) == pytest.approx(1.5)
    assert m.landweber_omega(np.array([2.0, 1.0]), 1.0) == pytest.approx(0.25)


def test_landweber_filter_by_hand(mini):
    sv = m.decompose(mini[0])[1]
    hand = {0: (0.0, 0.0, 0.0), 1: (1.0, 0.25, 0.01), 2: (1.0, 0.4375, 0.0199), 3: (1.0, 0.578125, 0.029701)}
    for k, f in hand.items():
        assert m.landweber_filter(sv, k, 1.0) == pytest.approx(f, abs=1e-12)


def test_landweber_filter_saturates_for_many_steps_and_diverges_beyond_two_over_sigma1_squared(mini):
    sv = m.decompose(mini[0])[1]
    assert m.landweber_filter(sv, 10 ** 4, 1.0) == pytest.approx([1.0, 1.0, 1.0 - 0.99 ** 10000])
    assert m.landweber_filter(sv, 3, 2.2)[0] == pytest.approx(1.0 + 1.2 ** 3)     # (1 - 2.2)^3 = -1.728: der Filter schießt über 1 hinaus


def test_landweber_first_step_is_the_transposed_matrix_times_the_data(mini):
    a, x, b, u, sv, vt, beta, c = _coords(mini)
    y = m.landweber_iterate(sv, beta, 1, 1.0)[0]
    assert vt.T @ y == pytest.approx(a.T @ b) and (vt.T @ y) == pytest.approx([1.33, 1.34, 1.33])


def test_landweber_many_steps_reach_the_direct_inversion(mini):
    a, x, b, u, sv, vt, beta, c = _coords(mini)
    assert vt.T @ m.landweber_iterate(sv, beta, 10 ** 5, 1.0)[0] == pytest.approx(x, abs=1e-6)


def test_cgls_plain_first_step_by_hand_and_third_step_is_the_solution(mini):
    a, x, b, u, sv, vt, beta, c = _coords(mini)
    ys = m.cgls_plain(sv, beta, 3)
    assert ys.shape == (4, 1, 3) and not ys[0].any()
    assert vt.T @ ys[1, 0] == pytest.approx(1.0000125 * np.array([1.33, 1.34, 1.33]), rel=1e-6)
    assert vt.T @ ys[2, 0] == pytest.approx(x, abs=1e-8)       # b hat nur zwei Komponenten: der Krylov-Raum ist nach zwei Schritten erschöpft
    assert vt.T @ ys[3, 0] == pytest.approx(x, abs=1e-8)


def test_cgls_exact_agrees_with_the_plain_recurrence_on_the_mini_instance_and_freezes_after_breakdown(mini):
    a, x, b, u, sv, vt, beta, c = _coords(mini)
    ex = m.cgls_exact(sv, beta, kmax=6)
    pl = m.cgls_plain(sv, beta, 3)
    assert ex.shape == (7, 1, 3)
    assert np.allclose(ex[:4], pl, atol=1e-10)
    assert np.allclose(ex[4], ex[3]) and np.allclose(ex[6], ex[3]) and vt.T @ ex[6, 0] == pytest.approx(x, abs=1e-8)


def test_cgls_exact_handles_a_zero_data_row(mini):
    sv = m.decompose(mini[0])[1]
    out = m.cgls_exact(sv, np.zeros((1, 3)), kmax=3)
    assert out.shape == (4, 1, 3) and not out.any()


def test_residual_and_solution_norm_by_hand(mini):
    a, x, b, u, sv, vt, beta, c = _coords(mini)
    zero = np.zeros((1, 1, 3))
    assert m.residual2(sv, beta, zero)[0, 0] == pytest.approx(float(b @ b))           # 1,69 + 1,96 + 1,69 = 5,34
    assert m.solution2(zero)[0, 0] == 0.0
    y = (c)[None, None, :]
    assert m.residual2(sv, beta, y)[0, 0] == pytest.approx(0.0, abs=1e-20)
    assert m.solution2(y)[0, 0] == pytest.approx(6.0)                                 # 1 + 4 + 1


def test_rel_errors_by_hand(mini):
    a, x, b, u, sv, vt, beta, c = _coords(mini)
    xn = float(np.linalg.norm(x))
    assert m.rel_errors(np.zeros((1, 3)), c, xn)[0] == pytest.approx(1.0)
    assert m.rel_errors(c[None, :], c, xn)[0] == pytest.approx(0.0, abs=1e-15)
    assert m.rel_errors((c + np.array([0.0, 0.0, 0.6]))[None, :], c, xn)[0] == pytest.approx(0.6 / np.sqrt(6.0))


def test_expected_error_by_hand(mini):
    a, x, b, u, sv, vt, beta, c = _coords(mini)
    xn = float(np.linalg.norm(x))
    ones, zeros = np.ones(3), np.zeros(3)
    assert m.expected_error(sv, c, zeros, 0.01, zeros, xn) == pytest.approx(1.0)                        # f = 0: nur der Bias, die ganze Wahrheit
    assert m.expected_error(sv, c, zeros, 0.01, ones, xn) == pytest.approx(0.01 * np.sqrt(1 + 4 + 100) / np.sqrt(6.0))
    rho = np.array([0.0, 0.0, 0.3])
    assert m.expected_error(sv, c, rho, 0.0, ones, xn) == pytest.approx(0.3 / np.sqrt(6.0))            # f = 1: der Modellfehler zählt als Bias


def test_discrepancy_stop_by_hand():
    res2 = np.array([[10.0, 0.5], [5.0, 0.4], [2.0, 0.3], [1.0, 0.2], [0.5, 0.1]])   # Zeilen k, Spalten Wiederholungen
    # sigma = 0,5, n = 4: Ziel (tau sigma)^2 n = 1,0; Spalte 2 liegt schon bei k = 0 darunter, zählt aber erst ab k = 1
    assert list(m.discrepancy_stop(res2, 0.5, 4)) == [3, 1]
    assert list(m.discrepancy_stop(res2, 0.5, 4, tau=2.0)) == [2, 1]
    assert list(m.discrepancy_stop(res2, 0.05, 4)) == [4, 4]                                             # nie erreicht: der letzte Index
    assert list(m.discrepancy_stop(res2, np.array([0.5, 0.05]), 4)) == [3, 4]                            # sigma je Zeile


def test_gcv_stop_by_hand():
    res2 = np.array([[8.0], [4.0], [1.5], [0.9], [0.1]])
    # G(k) = res2 / (4 - k)^2: k = 1: 0,444; k = 2: 0,375; k = 3: 0,9; k = 4: 0,1 / 1e-18 (n - k = 0 wird nach unten begrenzt)
    assert m.gcv_stop(res2, 4)[0] == 2


def test_lcurve_stop_finds_the_corner():
    rho = np.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -5.2, -5.4, -5.6, -5.8])       # log Residuum fällt, dann Plateau
    eta = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 2.5, 4.5, 6.5, 8.5])                 # log Lösungsnorm steigt erst langsam, dann steil
    res2 = np.exp(2 * np.concatenate([[0.5], rho]))[:, None]                           # Index 0 ist k = 0, die Ecke liegt bei rho = -5 (k = 6)
    sol2 = np.exp(2 * np.concatenate([[0.0], eta]))[:, None]
    assert m.lcurve_stop(res2, sol2)[0] == 6


def _polyline_curve(segs):
    """Kurve (log Residuum, log Loesungsnorm) aus Teilstuecken (Richtung in Grad, Schrittlaenge); gibt res2 und sol2 zurueck (Index 0 ist k = 0)."""
    import math

    pts = [(0.0, 0.0)]
    for ang, ln in segs:
        x, y = pts[-1]
        pts.append((x + ln * math.cos(math.radians(ang)), y + ln * math.sin(math.radians(ang))))
    rho = np.array([p[0] for p in pts])
    eta = np.array([p[1] for p in pts])
    return np.exp(2 * np.concatenate([[0.0], rho]))[:, None], np.exp(2 * np.concatenate([[0.0], eta]))[:, None]


def test_lcurve_stop_normalizes_by_the_speed_along_the_curve():
    """Zwei Biegungen: eine sanfte (30 Grad) mit Schrittlaenge 2, eine enge (90 Grad) mit Schrittlaenge 0,1. Der Zaehler der Kruemmung ist bei der sanften
    groesser (0,5 bis 1,0 gegen 0,005), die Kruemmung selbst bei der engen (14,1 gegen 0,14): gewaehlt wird die enge, k = 10."""
    res2, sol2 = _polyline_curve([(180, 2.0)] * 3 + [(150, 2.0)] * 3 + [(150, 0.1)] * 3 + [(60, 0.1)] * 3)
    assert m.lcurve_stop(res2, sol2)[0] == 10


def test_lcurve_stop_prefers_the_fast_sharp_bend_over_a_slow_almost_straight_one():
    """Eine fast gerade, langsame Strecke (1,5 Grad bei Schrittlaenge 0,1, Kruemmung 0,26) gegen eine 90-Grad-Biegung mit Schrittlaenge 3 (Kruemmung 0,33):
    die Biegung gewinnt, k = 10. Faengt falsche Potenzen und Vorzeichen im Nenner der Kruemmung."""
    res2, sol2 = _polyline_curve([(91.5, 0.1)] * 3 + [(90, 0.1)] * 3 + [(90, 3.0)] * 3 + [(0, 3.0)] * 3)
    assert m.lcurve_stop(res2, sol2)[0] == 10


def test_oracle_stop_is_the_argmin_per_column():
    err = np.array([[0.9, 0.8], [0.4, 0.7], [0.2, 0.3], [0.5, 0.1], [0.6, 0.2]])
    assert list(m.oracle_stop(err)) == [2, 3]
