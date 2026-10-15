"""Kern: Landweber-Filter, CGLS (mit voller Reorthogonalisierung und in der üblichen Rekurrenz), Abbruchregeln auf dem Iterationsindex. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_itr.py` gegen die explizite Iteration, CGLS in 100 Dezimalstellen (`decimal`), die Krylov-Definition und Monte Carlo.

Alles läuft in den Koordinaten der SVD A = U diag(sv) V^T: mit beta = U^T b und y = V^T x ist die Aufgabe diag(sv) y = beta. Die Iterierten von Landweber und
CGLS sind dieselben wie mit der Matrix selbst (U und V sind orthogonal); die Matrix-Vektor-Produkte werden nur gezählt.
"""

from __future__ import annotations

import numpy as np

import itr_constants as C

_TINY = 1e-300
_BREAKDOWN = 1e-12                      # relative Schwelle, ab der die Bidiagonalisierung als erschöpft gilt


def decompose(a: np.ndarray):
    """Volle SVD A = U diag(sv) V^T (sv absteigend)."""
    u, sv, vt = np.linalg.svd(a)
    return u, sv, vt


def landweber_omega(sv: np.ndarray, factor: float = 1.0) -> float:
    """Schrittweite factor / sigma_1^2; stabil für factor < 2."""
    return float(factor / sv[0] ** 2)


def landweber_filter(sv: np.ndarray, k, omega: float) -> np.ndarray:
    """Filterfaktoren nach k Schritten: f_i = 1 - (1 - omega sv_i^2)^k (k darf groß sein, auch 10^8)."""
    base = 1.0 - omega * np.asarray(sv, dtype=float) ** 2
    return 1.0 - np.power(base, float(round(k)))


def landweber_iterate(sv: np.ndarray, beta: np.ndarray, k, omega: float) -> np.ndarray:
    """y_k = f / sv * beta für beta der Form (reps, n) in SVD-Koordinaten."""
    return np.atleast_2d(beta) * (landweber_filter(sv, k, omega) / np.maximum(sv, _TINY))


def cgls_plain(sv: np.ndarray, beta: np.ndarray, kmax: int) -> np.ndarray:
    """CGLS in der üblichen Rekurrenz (doppelte Genauigkeit, ohne Reorthogonalisierung), vektorisiert über die Zeilen von beta; Form (kmax + 1, reps, n)."""
    s = np.asarray(sv, dtype=float)
    beta = np.atleast_2d(beta)
    y = np.zeros_like(beta)
    r = beta.copy()
    sr = s * r
    p = sr.copy()
    gam = (sr ** 2).sum(axis=1)
    out = [y.copy()]
    for _ in range(kmax):
        q = s * p
        alpha = gam / np.maximum((q ** 2).sum(axis=1), _TINY)
        y = y + alpha[:, None] * p
        r = r - alpha[:, None] * q
        sr = s * r
        gam_new = (sr ** 2).sum(axis=1)
        p = sr + (gam_new / np.maximum(gam, _TINY))[:, None] * p
        gam = gam_new
        out.append(y.copy())
    return np.array(out)


def cgls_exact(sv: np.ndarray, beta: np.ndarray, kmax: int = C.KMAX) -> np.ndarray:
    """CGLS ohne Rundungsstagnation: Golub-Kahan-Bidiagonalisierung (LSQR) mit voller Reorthogonalisierung; y_k löst min ||diag(sv) y - beta|| über den
    Krylov-Raum. Stimmt bis k = 42 auf 1e-14 mit CGLS in 100 Dezimalstellen überein (tests/test_oracle_itr.py). Form (kmax + 1, reps, n)."""
    s = np.asarray(sv, dtype=float)
    n = len(s)
    beta = np.atleast_2d(beta)
    out = np.zeros((kmax + 1, beta.shape[0], n))
    amat = np.diag(s)
    for r in range(beta.shape[0]):
        b1 = float(np.linalg.norm(beta[r]))
        if b1 == 0.0:
            continue
        us = [beta[r] / b1]
        v = amat.T @ us[0]
        a1 = float(np.linalg.norm(v))
        vs = [v / a1]
        alphas, betas = [a1], [b1]
        for k in range(1, kmax + 1):
            umat = np.array(us)
            w = amat @ vs[-1] - alphas[-1] * us[-1]
            w = w - umat.T @ (umat @ w)
            bn = float(np.linalg.norm(w))
            exhausted = bn <= _BREAKDOWN * b1     # Krylov-Raum erschöpft: B_k ist schon exakt, die Lösung ändert sich danach nicht mehr
            betas.append(0.0 if exhausted else bn)
            us.append(np.zeros(n) if exhausted else w / bn)
            bm = np.zeros((k + 1, k))
            for i in range(k):
                bm[i, i] = alphas[i]
                bm[i + 1, i] = betas[i + 1]
            rhs = np.zeros(k + 1)
            rhs[0] = b1
            z = np.linalg.lstsq(bm, rhs, rcond=None)[0]
            out[k, r] = np.array(vs[:k]).T @ z
            if exhausted:
                out[k + 1:, r] = out[k, r]
                break
            if k < kmax:
                vmat = np.array(vs)
                v = amat.T @ us[-1] - bn * vs[-1]
                v = v - vmat.T @ (vmat @ v)
                an = float(np.linalg.norm(v))
                if an <= _BREAKDOWN * b1:
                    out[k + 1:, r] = out[k, r]
                    break
                alphas.append(an)
                vs.append(v / an)
    return out


def residual2(sv: np.ndarray, beta: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """||diag(sv) y_k - beta||^2 je Iterierte und Zeile; ys der Form (K, reps, n), Rückgabe (K, reps)."""
    return ((np.asarray(sv) * ys - np.atleast_2d(beta)[None, :, :]) ** 2).sum(axis=-1)


def solution2(ys: np.ndarray) -> np.ndarray:
    """||y_k||^2 je Iterierte und Zeile."""
    return (ys ** 2).sum(axis=-1)


def rel_errors(ys: np.ndarray, c: np.ndarray, x_norm: float) -> np.ndarray:
    """||y_k - c|| / ||x|| (V ist orthogonal, also zählen die Koeffizienten c = V^T x); Form wie ys ohne letzte Achse."""
    return np.sqrt(((ys - c) ** 2).sum(axis=-1)) / x_norm


def expected_error(sv, c, rho, sigma, f, x_norm) -> float:
    """Wurzel von E||x_f - x||^2 / ||x|| für beliebige Filterfaktoren f: Bias f (c + rho) - c, Varianz f^2 sigma^2 / sv^2; rho = (U^T r) / sv."""
    bias = f * (c + rho) - c
    var = f ** 2 * sigma ** 2 / np.maximum(sv, _TINY) ** 2
    return float(np.sqrt(np.sum(bias ** 2 + var)) / x_norm)


# ----------------------------------------------------------------------------- Abbruchregeln auf dem Iterationsindex (Zeilen: Wiederholungen)

def discrepancy_stop(res2: np.ndarray, sigma, n: int, tau: float = 1.0) -> np.ndarray:
    """Erster Index k >= 1 mit Residuum^2 <= (tau sigma)^2 n; gibt es keinen, der letzte Index. res2 der Form (K, reps)."""
    target = (tau * np.asarray(sigma, dtype=float)) ** 2 * n
    ok = res2 <= (target.reshape(1, -1) if np.ndim(target) else target)
    ok[0] = False
    first = np.argmax(ok, axis=0)
    return np.where(ok.any(axis=0), first, res2.shape[0] - 1)


def gcv_stop(res2: np.ndarray, n: int) -> np.ndarray:
    """GCV mit df = k (Krylov-Dimension; für CGLS eine Näherung): Minimum von Residuum^2 / (n - k)^2 über k = 1 .. K - 1."""
    ks = np.arange(res2.shape[0])
    g = res2 / np.maximum(n - ks, 1e-9)[:, None] ** 2
    return np.argmin(g[1:], axis=0) + 1


def lcurve_stop(res2: np.ndarray, sol2: np.ndarray) -> np.ndarray:
    """Ecke der L-Kurve auf dem Iterationsindex: größte Krümmung von (log Residuum, log Lösungsnorm) über k (endliche Differenzen, innere Punkte k = 2 .. K - 2).
    Die Kurve läuft mit wachsendem k von der flachen Seite (Residuum fällt, Lösungsnorm kaum) auf die steile (Lösungsnorm wächst); das ist die umgekehrte Richtung
    wie bei wachsendem Strafparameter, deshalb steht der Zähler der Krümmung mit umgekehrtem Vorzeichen (sonst wählt die Regel die flachste Stelle, nicht die Ecke)."""
    rho = 0.5 * np.log(np.maximum(res2[1:], _TINY))
    eta = 0.5 * np.log(np.maximum(sol2[1:], _TINY))
    d1r, d1e = np.gradient(rho, axis=0), np.gradient(eta, axis=0)
    d2r, d2e = np.gradient(d1r, axis=0), np.gradient(d1e, axis=0)
    kap = (d2r * d1e - d1r * d2e) / np.maximum((d1r ** 2 + d1e ** 2) ** 1.5, _TINY)
    return np.argmax(kap[1:-1], axis=0) + 2


def oracle_stop(err: np.ndarray) -> np.ndarray:
    """Realisierungsorakel: der Index mit dem kleinsten tatsächlichen Fehler je Zeile; err der Form (K, reps)."""
    return np.argmin(err, axis=0)
