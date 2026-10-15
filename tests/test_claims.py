"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit."""

import numpy as np
import pytest

import itr_constants as C
import itr_evaluation as ev


@pytest.fixture(scope="module")
def halfconv():
    return ev.halfconv_experiment()


@pytest.fixture(scope="module")
def cost():
    return ev.cost_experiment()


@pytest.fixture(scope="module")
def stop():
    return ev.stop_experiment()


@pytest.fixture(scope="module")
def rounding():
    return ev.rounding_experiment()


# ----------------------------------------------------------------------------- Instanz und Presets

def test_standard_instance_numbers():
    a = ev.analyse(ev.Settings())
    assert 1.2e4 < a.cond < 1.3e4 and a.sigma == pytest.approx(3.4e-4, rel=0.02)
    assert a.err_iter == pytest.approx(0.1177, abs=0.002) and a.err_oracle == pytest.approx(0.1163, abs=0.002) and a.k_oracle == 28
    assert a.err_tik == pytest.approx(0.1168, abs=0.002) and a.lam_star == pytest.approx(0.0091, abs=0.0005)
    assert a.err_naive > 2.0


def test_preset_help_numbers():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    assert round(100 * res["Standard"].err_iter) == 12
    assert res["Landweber"].steps == 10000 and res["Landweber"].matvec == 20000 and res["Standard"].matvec == 60
    assert res["Landweber"].err_iter == pytest.approx(res["Standard"].err_iter, abs=0.002)
    assert round(100 * res["Zu spät"].err_iter) == 26 and res["Zu spät"].steps == 38
    assert round(100 * res["Rundung"].err_iter) == 12 and round(100 * res["Rundung"].curve_exact[40]) == 40
    assert res["Standard"].settings.method == "cgls" and res["Rundung"].settings.method == "cgls_plain"


# ----------------------------------------------------------------------------- Experiment 1

def test_semi_convergence_numbers(halfconv):
    r = halfconv
    assert r["k_star"] == 30 and r["err_star"] == pytest.approx(0.1157, abs=0.003)
    assert r["rms"][35] == pytest.approx(0.159, abs=0.006) and r["rms"][38] == pytest.approx(0.267, abs=0.01) and r["rms"][40] == pytest.approx(0.45, abs=0.012)
    assert r["rms"][1] == pytest.approx(0.307, abs=0.01) and r["rms"][5] == pytest.approx(0.147, abs=0.006) and r["rms"][10] == pytest.approx(0.142, abs=0.006)


def test_references_and_landweber_forgiveness(halfconv):
    r = halfconv
    assert r["tik_err"] == pytest.approx(0.1168, abs=0.002) and r["tsvd_k"] == 33 and r["tsvd_err"] == pytest.approx(0.1144, abs=0.002)
    assert 7000 < r["lw_best"] < 11000 and r["lw_err"] == pytest.approx(0.1159, abs=0.002)
    errs = [o["err"] for o in r["lw_offsets"]]
    assert max(errs) < 1.02 * r["lw_err"] and min(errs) >= r["lw_err"] - 1e-9          # von 0,5- bis 2,0-fach des besten Abbruchs: weniger als 1 % mehr
    by = {o["factor"]: o for o in r["offsets"]}
    assert by[0.7]["rms"] < 1.12 * r["err_star"] and by[1.4]["rms"] > 3.5 * r["err_star"] and by[1.4]["k"] == C.KMAX


# ----------------------------------------------------------------------------- Experiment 2

def test_cost_numbers(cost):
    rows = {r["noise"]: r for r in cost["rows"]}
    assert (rows[0.1]["cgls_k"], rows[0.1]["lw_k"]) == (3, 7)
    assert rows[1e-3]["cgls_k"] == 30 and 7000 < rows[1e-3]["lw_k"] < 11000
    assert rows[1e-3]["lw_mv"] > 100 * rows[1e-3]["cgls_mv"]
    for r in cost["rows"]:
        assert r["lw_err"] == pytest.approx(r["cgls_err"], abs=0.01) and r["lw_k"] > r["cgls_k"]
    lw = [r["lw_k"] for r in cost["rows"]]
    assert lw == sorted(lw)                                                    # weniger Rauschen, mehr Schritte


def test_step_size_changes_only_the_step_count(cost):
    errs = [o["err"] for o in cost["omegas"]]
    assert max(errs) - min(errs) < 1e-5
    prods = [o["k"] * o["factor"] for o in cost["omegas"]]
    assert max(prods) / min(prods) < 1.02
    assert cost["unstable"] > 10.0


# ----------------------------------------------------------------------------- Experiment 3

def test_stop_rule_numbers_honest(stop):
    d = stop["kinds"]["honest"]
    assert d["k_star"] == 30 and d["err_star"] == pytest.approx(0.1157, abs=0.003)
    r = d["rules"]
    assert r["dp"]["fail"] == 0.0 and r["dp"]["k_median"] == 24 and r["dp"]["median"] == pytest.approx(0.120, abs=0.004)
    assert 0.07 < r["gcv"]["fail"] < 0.18 and r["gcv"]["median"] < 0.13 and r["gcv"]["p90"] > 0.2
    assert 0.18 < r["lcurve"]["fail"] < 0.36 and 35 <= r["lcurve"]["k_median"] <= 38 and r["lcurve"]["median"] == pytest.approx(0.19, abs=0.02)
    assert r["oracle"]["fail"] == 0.0 and r["oracle"]["median"] == pytest.approx(0.1152, abs=0.003)


def test_stop_rule_numbers_crime(stop):
    d = stop["kinds"]["crime"]
    assert d["k_star"] == 31 and d["err_star"] == pytest.approx(0.0709, abs=0.003)
    r = d["rules"]
    assert r["dp"]["fail"] == 0.0 and 0.2 < r["gcv"]["fail"] < 0.4 and 0.45 < r["lcurve"]["fail"] < 0.7
    assert stop["kinds"]["crime"]["err_star"] < 0.7 * stop["kinds"]["honest"]["err_star"]


# ----------------------------------------------------------------------------- Experiment 4

def test_rounding_numbers(rounding):
    rows = {r["k"]: r for r in rounding["rows"]}
    assert rows[20]["plain"] == pytest.approx(rows[20]["exact"], abs=1e-6)
    assert rows[40]["exact"] == pytest.approx(0.45, abs=0.012) and rows[40]["plain"] == pytest.approx(0.116, abs=0.004)
    assert rows[33]["plain"] < rows[33]["exact"] - 0.005 and rows[36]["plain"] < 0.7 * rows[36]["exact"]
    assert rounding["k_plain"] == C.KMAX and rounding["plain_min"] == pytest.approx(0.1158, abs=0.003)


def test_all_claims_are_consistent_with_the_verdict_texts(halfconv, cost, stop, rounding):
    texts = [ev.verdict_halfconv(halfconv)[1], ev.verdict_cost(cost)[1], ev.verdict_stop(stop)[1], ev.verdict_rounding(rounding)[1]]
    assert "k = 30" in texts[0] and "45,1 %" in texts[0] and "Rang 33" in texts[0]
    assert "8.913" in texts[0] and "17.826" in texts[1] and "19.953" in texts[1] and "60" in texts[1]
    assert "k = 24" in texts[2] and "0 % gescheitert" in texts[2] and "L-Kurve bei k = 37" in texts[2]
    assert "45,1 %" in texts[3] and "11,6 %" in texts[3]
    assert np.isfinite(cost["unstable"])
