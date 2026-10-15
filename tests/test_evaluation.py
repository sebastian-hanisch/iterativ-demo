"""Auswertung: Formatierung, Analyse der Instanz, Verdikte (Art und Text), Form der Experimente und ihre Reproduzierbarkeit."""

import numpy as np
import pytest

import itr_constants as C
import itr_evaluation as ev


# ----------------------------------------------------------------------------- Formatierung

def test_de_sci_and_share():
    assert ev.de(-1.234, 2) == "−1,23" and ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.share(0.1525) == "15 %"


def test_err_pct_digits_by_magnitude():
    assert [ev.err_pct(v) for v in (2.58, 0.1188, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,9 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]


def test_int_text_and_noise_label():
    assert [ev.int_text(v) for v in (7, 8913, 12345678)] == ["7", "8.913", "1,2·10⁷"]
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0 %", "0,0001 %", "0,001 %", "0,01 %", "0,1 %", "1 %", "10 %"]


def test_landweber_steps_are_powers_of_ten_of_the_slider():
    assert [ev.landweber_steps(v) for v in (0, 5, 10, 20, 40)] == [0, 10, 100, 10000, 10 ** 8]


# ----------------------------------------------------------------------------- Analyse der Instanz

def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_the_chosen_iterate_matches_the_method_and_the_step_count():
    a = ev.analyse(ev.Settings(method="cgls", k=12))
    assert a.steps == 12 and a.matvec == 24 and a.err_iter == pytest.approx(a.curve_exact[12])
    p = ev.analyse(ev.Settings(method="cgls_plain", k=12))
    assert p.err_iter == pytest.approx(p.curve_plain[12])
    lw = ev.analyse(ev.Settings(method="landweber", k=10))
    assert lw.steps == 100 and lw.matvec == 200 and np.isfinite(lw.lam_equiv)
    zero = ev.analyse(ev.Settings(method="landweber", k=0))
    assert zero.steps == 0 and not zero.x_iter.any() and zero.err_iter == pytest.approx(1.0)


def test_k_zero_is_the_zero_solution_for_every_method():
    for m in C.METHODS:
        a = ev.analyse(ev.Settings(method=m, k=0))
        assert a.steps == 0 and not a.x_iter.any()


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b, c = ev.analyse(ev.Settings()), ev.analyse(ev.Settings()), ev.analyse(ev.Settings(seed=12))
    assert np.array_equal(a.x_iter, b.x_iter) and a.picks == b.picks and not np.array_equal(a.b_beta, c.b_beta)


def test_the_oracle_is_the_minimum_of_the_curve_and_no_rule_beats_it():
    a = ev.analyse(ev.Settings())
    assert a.err_oracle == pytest.approx(float(a.curve_exact[1:].min())) and a.curve_exact.shape == a.curve_plain.shape == (C.KMAX + 1,)
    assert a.err_oracle <= min(a.pick_errors[k] for k in ("dp", "gcv", "lcurve")) + 1e-15
    assert a.picks["oracle"] == a.k_oracle


def test_data_kinds_differ_and_crime_is_easier():
    h, c = ev.analyse(ev.Settings(data="ehrlich")), ev.analyse(ev.Settings(data="verbrechen"))
    assert c.err_oracle < h.err_oracle


def test_the_singular_flag_follows_the_condition_number():
    assert ev.analyse(ev.Settings(n=96, w=0.05)).singular and not ev.analyse(ev.Settings()).singular


def test_small_grids_and_all_corners_are_finite():
    for n in (12, 24, 96):
        for w in (0.01, 0.05):
            for noise in (0.0, 1e-1):
                for m, k in (("cgls", 40), ("landweber", 40), ("cgls_plain", 40)):
                    a = ev.analyse(ev.Settings(n=n, w=w, noise=noise, method=m, k=k))
                    assert np.isfinite(a.curve_exact).all() and np.isfinite(a.err_iter) and np.isfinite(a.lw_curve).all()


# ----------------------------------------------------------------------------- Verdikte

def test_verdict_kinds_and_texts_of_the_presets():
    out = {name: ev.verdict_instance(ev.analyse(ev.Settings(**p))) for name, p in C.PRESETS.items()}
    assert out["Standard"][0] == "success" and out["Zu spät"][0] == "warning"
    assert "mehr als das Doppelte" in out["Zu spät"][1] and "zu spät" in out["Zu spät"][1] and "Tikhonov" in out["Standard"][1]


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_halfconv_structure_and_reproducibility():
    r1, r2 = ev.halfconv_experiment(reps=40), ev.halfconv_experiment(reps=40)
    assert np.array_equal(r1["rms"], r2["rms"]) and r1["rms"].shape == (C.KMAX + 1,) and len(r1["offsets"]) == len(C.EXP_OFFSETS)
    assert r1["rms"][r1["k_star"]] == pytest.approx(r1["rms"][1:].min())


def test_cost_structure():
    r = ev.cost_experiment(reps=30)
    assert [x["noise"] for x in r["rows"]] == list(C.EXP_NOISE_LEVELS) and [o["factor"] for o in r["omegas"]] == list(C.EXP_OMEGA_FACTORS)


def test_stop_structure_and_reproducibility():
    r1, r2 = ev.stop_experiment(reps=40), ev.stop_experiment(reps=40)
    assert set(r1["kinds"]) == {"honest", "crime"} and set(r1["kinds"]["honest"]["rules"]) == {"dp", "gcv", "lcurve", "oracle"}
    assert r1["kinds"]["honest"]["rules"]["gcv"]["rms"] == r2["kinds"]["honest"]["rules"]["gcv"]["rms"]
    o = r1["kinds"]["honest"]["rules"]["oracle"]
    assert o["fail"] == 0.0 and o["median"] <= r1["kinds"]["honest"]["rules"]["dp"]["median"] + 1e-12


def test_rounding_structure():
    r = ev.rounding_experiment(reps=20)
    assert [x["k"] for x in r["rows"]] == list(C.EXP_PLAIN_KS) and r["exact"].shape == r["plain"].shape == (C.KMAX + 1,)
