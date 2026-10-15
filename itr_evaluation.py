"""Auswertung: Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import itr_constants as C
import itr_model as model
from inverse_generator import SplitMix64, blurred_exact, grid, noise_sigma, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def int_text(k: float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner; ab einer Million als Zehnerpotenz."""
    k = int(round(k))
    return f"{k:,}".replace(",", ".") if k < 10 ** 6 else sci(float(k))


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def share(x: float) -> str:
    """Anteil (0 bis 1) als ganze Prozent."""
    return f"{de(100.0 * x, 0)} %"


def landweber_steps(slider: int) -> int:
    """Zahl der Landweber-Schritte zum Reglerwert: 0 bleibt 0, sonst 10^(Regler/5)."""
    return 0 if slider <= 0 else int(round(10.0 ** (slider / C.LANDWEBER_DECADE)))


# ----------------------------------------------------------------------------- Welt

@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    data: str = "ehrlich"
    method: str = "cgls"
    k: int = C.DEFAULT_K
    seed: int = C.DEFAULT_SEED


class World:
    """Matrix, Zerlegung, Wahrheit, Modellfehler und alle Auswertungen der Iterationsverfahren."""

    def __init__(self, n: int, w: float, noise: float | None = None, sigma: float | None = None):
        self.n, self.w = n, w
        self.a = operator(n, w)
        self.u, self.sv, self.vt = model.decompose(self.a)
        self.x = phantom(grid(n))
        self.g = blurred_exact(grid(n), w)
        self.c = self.vt @ self.x
        self.rho_honest = (self.u.T @ (self.g - self.a @ self.x)) / np.maximum(self.sv, 1e-300)
        self.sigma = float(sigma) if sigma is not None else noise_sigma(n, w, noise)
        self.x_norm = float(np.linalg.norm(self.x))
        self.cond = float(self.sv[0] / self.sv[-1]) if self.sv[-1] > 0 else math.inf
        self.omega = model.landweber_omega(self.sv)

    def rho(self, honest: bool) -> np.ndarray:
        return self.rho_honest if honest else np.zeros(self.n)

    def beta(self, honest: bool, reps: int, seed: int) -> np.ndarray:
        base = self.g if honest else self.a @ self.x
        e = SplitMix64(seed).normal(self.n * reps).reshape(reps, self.n) * self.sigma
        return (base[None, :] + e) @ self.u

    def landweber_expected(self, k, honest: bool, omega: float | None = None) -> float:
        om = self.omega if omega is None else omega
        return model.expected_error(self.sv, self.c, self.rho(honest), self.sigma, model.landweber_filter(self.sv, k, om), self.x_norm)

    def landweber_best(self, honest: bool, omega: float | None = None) -> tuple[int, float]:
        lo, hi, pts = C.LW_GRID
        ks = np.unique(np.round(np.logspace(lo, hi, pts)).astype(int))
        e = np.array([self.landweber_expected(k, honest, omega) for k in ks])
        i = int(np.argmin(e))
        return int(ks[i]), float(e[i])

    def tikhonov_best(self, honest: bool) -> tuple[float, float]:
        lo, hi, pts = C.LAM_GRID
        lams = np.logspace(math.log10(lo), math.log10(hi), pts)
        e = np.array([model.expected_error(self.sv, self.c, self.rho(honest), self.sigma, self.sv ** 2 / (self.sv ** 2 + l ** 2), self.x_norm) for l in lams])
        i = int(np.argmin(e))
        return float(lams[i]), float(e[i])

    def tsvd_best(self, honest: bool) -> tuple[int, float]:
        e = []
        for k in range(self.n + 1):
            f = np.zeros(self.n)
            f[:k] = 1.0
            e.append(model.expected_error(self.sv, self.c, self.rho(honest), self.sigma, f, self.x_norm))
        i = int(np.argmin(e))
        return i, float(e[i])


_BUNDLES: dict = {}


def bundle(honest: bool, reps: int, seed: int, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE, sigma: float | None = None) -> dict:
    """CGLS (exakt und übliche Rekurrenz) über feste Ströme: Iterierte, Residuen, Lösungsnormen und Fehler; zwischengespeichert, weil mehrere Experimente sie teilen."""
    key = (honest, reps, seed, n, w, noise, sigma)
    if key not in _BUNDLES:
        wd = World(n, w, noise, sigma)
        beta = wd.beta(honest, reps, seed)
        ex = model.cgls_exact(wd.sv, beta)
        pl = model.cgls_plain(wd.sv, beta, C.KMAX)
        _BUNDLES[key] = {"wd": wd, "beta": beta, "exact": ex, "plain": pl, "err_exact": model.rel_errors(ex, wd.c, wd.x_norm), "err_plain": model.rel_errors(pl, wd.c, wd.x_norm),
                         "res2": model.residual2(wd.sv, beta, ex), "sol2": model.solution2(ex)}
    return _BUNDLES[key]


def rms(e: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(e ** 2, axis=axis))


def summarize(errs: np.ndarray, opt: float) -> dict:
    return {"rms": float(np.sqrt(np.mean(errs ** 2))), "median": float(np.median(errs)), "p90": float(np.quantile(errs, 0.9)),
            "fail": float(np.mean(errs > C.FAIL_FACTOR * opt))}


# ----------------------------------------------------------------------------- Instanz

@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    sv: np.ndarray
    cond: float
    sigma: float
    b_beta: np.ndarray
    steps: int               # Zahl der Iterationen des gewählten Verfahrens
    matvec: int              # Matrix-Vektor-Produkte (A und Aᵀ je Schritt)
    x_iter: np.ndarray       # gewählte Iterierte
    x_exact_best: np.ndarray # CGLS exakt beim besten Abbruch dieser Stichprobe (Orakel)
    x_tik: np.ndarray        # Tikhonov beim besten erwarteten Parameter (Vergleich)
    x_naive: np.ndarray
    err_iter: float
    err_oracle: float        # bester Fehler dieser Stichprobe über die CGLS-Iterierten
    k_oracle: int
    err_tik: float           # erwarteter Fehler von Tikhonov beim besten λ
    lam_star: float
    err_naive: float
    lam_equiv: float         # Landweber: 1 / sqrt(omega k), die entsprechende Tikhonov-Größe
    curve_exact: np.ndarray  # Fehler dieser Stichprobe je k (CGLS exakt), Länge KMAX + 1
    curve_plain: np.ndarray
    lw_steps: np.ndarray     # Gitter der Landweber-Schritte
    lw_curve: np.ndarray     # erwarteter Landweber-Fehler über das Gitter
    picks: dict              # Abbruchregeln auf CGLS exakt: Index je Regel
    pick_errors: dict
    x_norm: float
    singular: bool


def analyse(settings: Settings) -> Analysis:
    st = settings
    honest = st.data != "verbrechen"
    wd = World(st.n, st.w, st.noise)
    beta = wd.beta(honest, 1, st.seed)
    ex = model.cgls_exact(wd.sv, beta)
    pl = model.cgls_plain(wd.sv, beta, C.KMAX)
    err_ex = model.rel_errors(ex[:, 0, :], wd.c, wd.x_norm)
    err_pl = model.rel_errors(pl[:, 0, :], wd.c, wd.x_norm)
    k = max(0, min(int(st.k), C.K_MAX))
    if st.method == "landweber":
        steps = landweber_steps(k)
        y = model.landweber_iterate(wd.sv, beta, steps, wd.omega)[0]
        matvec = 2 * steps
        lam_eq = 1.0 / math.sqrt(wd.omega * steps) if steps > 0 else math.inf
    else:
        steps = k
        y = (pl if st.method == "cgls_plain" else ex)[k, 0]
        matvec = 2 * steps
        lam_eq = math.nan
    xi = wd.vt.T @ y
    lo, hi, pts = C.LW_GRID
    lw_steps = np.unique(np.round(np.logspace(lo, hi, pts)).astype(int))
    lw_curve = np.array([wd.landweber_expected(int(kk), honest) for kk in lw_steps])
    lam_star, err_tik = wd.tikhonov_best(honest)
    f_tik = wd.sv ** 2 / (wd.sv ** 2 + lam_star ** 2)
    x_tik = wd.vt.T @ (beta[0] * f_tik / wd.sv)
    k_or = int(np.argmin(err_ex[1:])) + 1
    res2 = model.residual2(wd.sv, beta, ex)
    sol2 = model.solution2(ex)
    picks = {"dp": int(model.discrepancy_stop(res2, wd.sigma, st.n)[0]), "gcv": int(model.gcv_stop(res2, st.n)[0]), "lcurve": int(model.lcurve_stop(res2, sol2)[0]), "oracle": k_or}
    return Analysis(
        settings=st, s=grid(st.n), x=wd.x, sv=wd.sv, cond=wd.cond, sigma=wd.sigma, b_beta=beta[0], steps=steps, matvec=matvec, x_iter=xi,
        x_exact_best=wd.vt.T @ ex[k_or, 0], x_tik=x_tik, x_naive=wd.vt.T @ (beta[0] / wd.sv), err_iter=float(model.rel_errors(y, wd.c, wd.x_norm)),
        err_oracle=float(err_ex[k_or]), k_oracle=k_or, err_tik=err_tik, lam_star=lam_star,
        err_naive=float(model.rel_errors(beta[0] / wd.sv, wd.c, wd.x_norm)), lam_equiv=lam_eq, curve_exact=err_ex, curve_plain=err_pl, lw_steps=lw_steps, lw_curve=lw_curve,
        picks=picks, pick_errors={kk: float(err_ex[v]) for kk, v in picks.items()}, x_norm=wd.x_norm, singular=bool(wd.cond >= C.SINGULAR_COND),
    )


# ----------------------------------------------------------------------------- Experiment 1: Halbkonvergenz und Nachsicht

def halfconv_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    bd = bundle(True, reps, C.SEED_BASE + 1, n, w, noise)
    wd = bd["wd"]
    r = rms(bd["err_exact"], axis=1)
    med = np.median(bd["err_exact"], axis=1)
    k_star = int(np.argmin(r[1:])) + 1
    offs = []
    for f in C.EXP_OFFSETS:
        kk = min(max(1, int(round(k_star * f))), C.KMAX)
        offs.append({"factor": f, "k": kk, "rms": float(r[kk])})
    kl, el = wd.landweber_best(True)
    lw_offs = [{"factor": f, "k": max(1, int(round(kl * f))), "err": wd.landweber_expected(max(1, int(round(kl * f))), True)} for f in C.EXP_OFFSETS]
    lam, et = wd.tikhonov_best(True)
    kt, ets = wd.tsvd_best(True)
    return {"k": np.arange(C.KMAX + 1), "rms": r, "median": med, "k_star": k_star, "err_star": float(r[k_star]), "offsets": offs, "lw_best": kl, "lw_err": el, "lw_offsets": lw_offs,
            "tik_lam": lam, "tik_err": et, "tsvd_k": kt, "tsvd_err": ets, "reps": reps, "n": n, "naive": wd.landweber_expected(10 ** 12, True)}


# ----------------------------------------------------------------------------- Experiment 2: was es kostet

def cost_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W) -> dict:
    rows = []
    for i, nz in enumerate(C.EXP_NOISE_LEVELS):
        bd = bundle(True, reps, C.SEED_BASE + 10 + i, n, w, nz)
        wd = bd["wd"]
        r = rms(bd["err_exact"], axis=1)
        kc = int(np.argmin(r[1:])) + 1
        kl, el = wd.landweber_best(True)
        lam, et = wd.tikhonov_best(True)
        rows.append({"noise": nz, "cgls_k": kc, "cgls_err": float(r[kc]), "lw_k": kl, "lw_err": el, "cgls_mv": 2 * kc, "lw_mv": 2 * kl, "tik_err": et, "lam": lam,
                     "equiv": kl * lam ** 2 * wd.sv[0] ** 2})
    wd = World(n, w, C.DEFAULT_NOISE)
    omegas = []
    for f in C.EXP_OMEGA_FACTORS:
        om = model.landweber_omega(wd.sv, f)
        k, e = wd.landweber_best(True, om)
        omegas.append({"factor": f, "k": k, "err": e})
    return {"rows": rows, "omegas": omegas, "unstable": wd.landweber_expected(50, True, model.landweber_omega(wd.sv, 2.1)), "n": n, "reps": reps}


# ----------------------------------------------------------------------------- Experiment 3: wann stoppen?

def stop_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    out = {"kinds": {}, "reps": reps, "n": n}
    for kind, honest, seed in (("honest", True, C.SEED_BASE + 1), ("crime", False, C.SEED_BASE + 2)):
        bd = bundle(honest, reps, seed, n, w, noise)
        wd = bd["wd"]
        r = rms(bd["err_exact"], axis=1)
        kc = int(np.argmin(r[1:])) + 1
        opt = float(r[kc])
        picks = {"dp": model.discrepancy_stop(bd["res2"], wd.sigma, n), "gcv": model.gcv_stop(bd["res2"], n), "lcurve": model.lcurve_stop(bd["res2"], bd["sol2"]),
                 "oracle": model.oracle_stop(bd["err_exact"])}
        rules = {}
        for key, idx in picks.items():
            e = bd["err_exact"][idx, np.arange(reps)]
            rules[key] = {**summarize(e, opt), "k_median": float(np.median(idx)), "k_q": np.quantile(idx, [0.05, 0.95])}
        out["kinds"][kind] = {"k_star": kc, "err_star": opt, "rules": rules}
    return out


# ----------------------------------------------------------------------------- Experiment 4: die Rundung lügt

def rounding_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    bd = bundle(True, reps, C.SEED_BASE + 1, n, w, noise)
    re, rp = rms(bd["err_exact"], axis=1), rms(bd["err_plain"], axis=1)
    rows = [{"k": k, "exact": float(re[k]), "plain": float(rp[k])} for k in C.EXP_PLAIN_KS]
    return {"rows": rows, "k": np.arange(C.KMAX + 1), "exact": re, "plain": rp, "k_exact": int(np.argmin(re[1:])) + 1, "k_plain": int(np.argmin(rp[1:])) + 1,
            "plain_min": float(rp[1:].min()), "exact_min": float(re[1:].min()), "reps": reps, "n": n}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    st = a.settings
    rel = a.err_iter / a.err_oracle if a.err_oracle > 0 else math.inf
    lead = f"{C.METHOD_SHORT[st.method]} nach {int_text(a.steps)} Schritten ({int_text(a.matvec)} Matrix-Vektor-Produkte): Fehler {err_pct(a.err_iter)}."
    tail = f" Der beste Abbruch dieser Stichprobe wäre CGLS nach {a.k_oracle} Schritten mit {err_pct(a.err_oracle)}; Tikhonov beim besten λ hat erwartet {err_pct(a.err_tik)}."
    if a.err_iter > C.FAIL_FACTOR * a.err_oracle:
        side = "zu früh" if (st.method != "landweber" and a.steps < a.k_oracle) or (st.method == "landweber" and a.steps < 1000) else "zu spät"
        return "warning", f"{lead} Das ist mehr als das Doppelte des besten Abbruchs ({side} gestoppt).{tail}"
    return "success", f"{lead} Das liegt beim {de(rel, 2)}-Fachen des besten Abbruchs.{tail}"


def verdict_halfconv(res: dict) -> tuple[str, str]:
    by = {o["factor"]: o for o in res["offsets"]}
    lo, hi = by[0.7], by[1.4]
    return "info", (f"CGLS hat sein Minimum bei k = {res['k_star']} mit {err_pct(res['err_star'])}; bei k = {C.KMAX} sind es {err_pct(res['rms'][C.KMAX])}. Landweber braucht "
                    f"{int_text(res['lw_best'])} Schritte für {err_pct(res['lw_err'])}. Zum Vergleich TSVD (Rang {res['tsvd_k']}) {err_pct(res['tsvd_err'])} und "
                    f"Tikhonov {err_pct(res['tik_err'])}. Zu früh (k = {lo['k']}) kostet bei CGLS {err_pct(lo['rms'])}, zu spät (k = {hi['k']}) {err_pct(hi['rms'])}: "
                    "die Strafe ist nicht symmetrisch.")


def verdict_cost(res: dict) -> tuple[str, str]:
    mid = next(r for r in res["rows"] if abs(r["noise"] - C.DEFAULT_NOISE) < 1e-12)
    lo, hi = res["rows"][0], res["rows"][-1]
    return "info", (f"Bei {noise_label(mid['noise'])} Rauschen braucht Landweber {int_text(mid['lw_mv'])} Matrix-Vektor-Produkte, CGLS {int_text(mid['cgls_mv'])}: der Fehler ist gleich "
                    f"({err_pct(mid['lw_err'])} und {err_pct(mid['cgls_err'])}). Von {noise_label(lo['noise'])} auf {noise_label(hi['noise'])} Rauschen wächst Landweber von "
                    f"{int_text(lo['lw_k'])} auf {int_text(hi['lw_k'])} Schritte, CGLS von {lo['cgls_k']} auf {hi['cgls_k']}.")


def verdict_stop(res: dict) -> tuple[str, str]:
    h = res["kinds"]["honest"]["rules"]
    return "warning", (f"Ehrliche Daten, {res['reps']} feste Wiederholungen: Das Diskrepanzprinzip stoppt im Median bei k = {int(h['dp']['k_median'])} mit {err_pct(h['dp']['median'])} "
                       f"({share(h['dp']['fail'])} gescheitert), GCV bei k = {int(h['gcv']['k_median'])} mit {err_pct(h['gcv']['median'])} und {share(h['gcv']['fail'])} "
                       f"Katastrophen, die L-Kurve bei k = {int(h['lcurve']['k_median'])} mit {err_pct(h['lcurve']['median'])} und {share(h['lcurve']['fail'])} gescheitert. "
                       f"Das Orakel erreicht {err_pct(h['oracle']['median'])}.")


def verdict_rounding(res: dict) -> tuple[str, str]:
    r40 = next(r for r in res["rows"] if r["k"] == C.KMAX)
    return "warning", (f"Bei k = {C.KMAX} liegt der Fehler der exakten Iteration bei {err_pct(r40['exact'])}, der der üblichen Rekurrenz in doppelter Genauigkeit bei {err_pct(r40['plain'])}. "
                       f"Die übliche Rekurrenz steigt bis k = {C.KMAX} nicht wieder an (Minimum {err_pct(res['plain_min'])}): die Rundung wirkt als zusätzliche "
                       "Regularisierung und verbirgt die Halbkonvergenz, die in exakter Arithmetik gälte.")
