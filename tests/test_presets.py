"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import itr_constants as C
import itr_evaluation as ev
import itr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["data"] in C.DATA_CHOICES and p["method"] in C.METHODS, name
        assert C.K_MIN <= p["k"] <= C.K_MAX and (p["k"] - C.K_MIN) % C.K_STEP == 0 and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, lw, late, rnd = (res[k] for k in C.PRESETS)
    assert std.err_iter < 0.13 and std.err_iter < 1.05 * std.err_oracle
    assert lw.steps == 10 ** 4 and abs(lw.err_iter - std.err_iter) < 0.003
    assert late.err_iter > 2 * late.err_oracle and 0.24 < late.err_iter < 0.28
    assert rnd.err_iter < 0.13 and rnd.curve_exact[40] > 0.35


def test_experiment_constants_are_consistent():
    assert C.EXP_REPS >= 200 and C.FAIL_FACTOR == 2.0 and C.DEFAULT_NOISE in C.EXP_NOISE_LEVELS and 1.0 in C.EXP_OMEGA_FACTORS and max(C.EXP_OMEGA_FACTORS) < 2.0
    assert max(C.EXP_PLAIN_KS) == C.KMAX == C.K_MAX and 1.0 in C.EXP_OFFSETS and C.LANDWEBER_DECADE * 8 == C.K_MAX
