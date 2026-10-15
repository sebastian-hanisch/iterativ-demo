"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import itr_constants as C
from itr_evaluation import Analysis, de, err_pct, int_text, noise_label, sci

ERR_TICKS = dict(tickvals=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 20], ticktext=["5 %", "10 %", "20 %", "50 %", "100 %", "200 %", "500 %", "2000 %"])
RULES = ("dp", "gcv", "lcurve", "oracle")
RULE_SHORT = {"dp": "Diskrepanzprinzip", "gcv": "GCV", "lcurve": "L-Kurve", "oracle": "Orakel"}
RULE_COLORS = {"dp": "#4c78a8", "gcv": "#D68A2E", "lcurve": "#8e44ad", "oracle": "#8a8a8a"}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Punkt": [f"{i + 1}" for i in range(k)],
        "Ort s": a.s[:k],
        "Wahrheit x": a.x[:k],
        "Gewählte Iterierte": a.x_iter[:k],
        "Bester CGLS-Abbruch": a.x_exact_best[:k],
        "Tikhonov (bestes λ)": a.x_tik[:k],
    })


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 3)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, gewählte Iterierte, bester CGLS-Abbruch dieser Stichprobe, Tikhonov und (blass) die direkte Inversion; die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    st = a.settings
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x_naive, mode="lines", name="direkte Inversion", line=dict(color=C.COLOR_NAIVE, width=1, dash="dot"), opacity=0.5, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_tik, mode="lines", name=f"Tikhonov, bestes λ = {sci(a.lam_star)}", line=dict(color=C.COLOR_TIK, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_exact_best, mode="lines", name=f"bester CGLS-Abbruch dieser Stichprobe (k = {a.k_oracle})", line=dict(color=C.COLOR_GREY, width=2, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_iter, mode="lines+markers", name=f"{C.METHOD_SHORT[st.method]}, {int_text(a.steps)} Schritte", line=dict(color=C.METHOD_COLORS[st.method], width=2.5),
                             marker=dict(size=5), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    peak = float(np.max(np.abs(a.x_iter))) if np.all(np.isfinite(a.x_iter)) else float("inf")
    note = ""
    if peak > max(abs(lo), abs(hi)):
        pt = "unendlich" if not np.isfinite(peak) else (de(peak, 1) if peak < 1000.0 else sci(peak))
        note = f"Die gewählte Iterierte reicht bis ±{pt} und wird bei {de(lo, 1)} bis {de(hi, 1)} abgeschnitten."
    return _fixed(fig, 380, title=dict(text=note, font=dict(size=12, color="#9aa0a6"), x=0.0))


def build_semiconv_chart(a: Analysis) -> go.Figure:
    """Fehler dieser Stichprobe über den CGLS-Index: exakte Iteration gegen die übliche Rekurrenz in doppelter Genauigkeit."""
    ks = np.arange(C.KMAX + 1)
    fig = go.Figure()
    fig.add_hline(y=a.err_tik, line=dict(color=C.COLOR_TIK, dash="dot"), annotation_text="Tikhonov, bestes λ (erwartet)", annotation_position="top left")
    fig.add_trace(go.Scatter(x=ks, y=a.curve_exact, mode="lines+markers", name="CGLS exakt", line=dict(color=C.COLOR_CGLS, width=3), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=ks, y=a.curve_plain, mode="lines", name="CGLS übliche Rekurrenz", line=dict(color=C.COLOR_PLAIN, width=2, dash="dash")))
    if a.settings.method != "landweber":
        k = min(a.steps, C.KMAX)
        y = a.curve_plain[k] if a.settings.method == "cgls_plain" else a.curve_exact[k]
        fig.add_trace(go.Scatter(x=[k], y=[y], mode="markers", name=f"gewählt: k = {k}", marker=dict(size=15, color=C.METHOD_COLORS[a.settings.method], line=dict(width=2, color="#ffffff"))))
    fig.add_trace(go.Scatter(x=[a.k_oracle], y=[a.err_oracle], mode="markers", name=f"bester Abbruch: k = {a.k_oracle}",
                             marker=dict(size=13, color=C.COLOR_GREY, symbol="diamond", line=dict(width=2, color="#ffffff"))))
    fig.update_xaxes(title="Iteration k")
    fig.update_yaxes(title="Fehler relativ zur Signalnorm", type="log", range=[-1.2, 1.0], **ERR_TICKS)
    return _fixed(fig, 400)


def build_landweber_chart(a: Analysis) -> go.Figure:
    """Erwarteter Fehler von Landweber über die Schrittzahl (logarithmische Achse) im Vergleich zu CGLS und Tikhonov."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.lw_steps, y=a.lw_curve, mode="lines", name="Landweber, erwarteter Fehler", line=dict(color=C.COLOR_LW, width=3)))
    fig.add_hline(y=a.err_tik, line=dict(color=C.COLOR_TIK, dash="dot"), annotation_text="Tikhonov, bestes λ", annotation_position="top left")
    if a.settings.method == "landweber" and a.steps > 0:
        i = int(np.argmin(np.abs(np.log10(a.lw_steps) - np.log10(a.steps))))
        fig.add_trace(go.Scatter(x=[a.lw_steps[i]], y=[a.lw_curve[i]], mode="markers", name=f"gewählt: {int_text(a.steps)} Schritte",
                                 marker=dict(size=15, color=C.COLOR_LW, line=dict(width=2, color="#ffffff"))))
    fig.update_xaxes(title="Landweber-Schritte", type="log", tickformat=".0e")
    fig.update_yaxes(title="Erwarteter Fehler relativ zur Signalnorm", type="log", range=[-1.2, 1.0], **ERR_TICKS)
    return _fixed(fig, 380)


def build_halfconv_chart(res: dict) -> go.Figure:
    ks = res["k"][1:]
    fig = go.Figure()
    fig.add_hline(y=res["tik_err"], line=dict(color=C.COLOR_TIK, dash="dot"), annotation_text="Tikhonov, bestes λ", annotation_position="top left")
    fig.add_hline(y=res["lw_err"], line=dict(color=C.COLOR_LW, dash="dash"), annotation_text="Landweber, bester Abbruch", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=ks, y=res["rms"][1:], mode="lines+markers", name="CGLS, RMS-Fehler", line=dict(color=C.COLOR_CGLS, width=3), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=ks, y=res["median"][1:], mode="lines", name="CGLS, Median", line=dict(color=C.COLOR_CGLS, width=1.5, dash="dot")))
    fig.update_xaxes(title="Iteration k")
    fig.update_yaxes(title="Fehler relativ zur Signalnorm", type="log", range=[-1.2, 0.8], **ERR_TICKS)
    return _fixed(fig, 380)


def build_cost_chart(res: dict) -> go.Figure:
    xs = [noise_label(r["noise"]) for r in res["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["cgls_mv"] for r in res["rows"]], name="CGLS", marker_color=C.COLOR_CGLS))
    fig.add_trace(go.Bar(x=xs, y=[r["lw_mv"] for r in res["rows"]], name="Landweber", marker_color=C.COLOR_LW))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rauschen (% des Effektivwerts der Daten)", type="category")
    fig.update_yaxes(title="Matrix-Vektor-Produkte bis zum besten Abbruch", type="log", tickvals=[10, 100, 1000, 10000, 100000], ticktext=["10", "100", "1.000", "10.000", "100.000"])
    return _fixed(fig, 340)


def build_stop_chart(res: dict, kind: str) -> go.Figure:
    d = res["kinds"][kind]
    fig = go.Figure()
    for key in RULES:
        v = d["rules"][key]
        fig.add_trace(go.Bar(x=[RULE_SHORT[key]], y=[100.0 * v["median"]], name=RULE_SHORT[key], marker_color=RULE_COLORS[key],
                             error_y=dict(type="data", symmetric=False, array=[100.0 * (v["p90"] - v["median"])], arrayminus=[0.0], color="#555555"),
                             text=[err_pct(v["median"])], textposition="outside", hovertemplate="Median %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=100.0 * d["err_star"], line=dict(color="#888888", dash="dash"), annotation_text="erwartetes Optimum", annotation_position="top left")
    fig.add_hline(y=100.0 * C.FAIL_FACTOR * d["err_star"], line=dict(color=C.COLOR_NAIVE, dash="dot"), annotation_text="Doppeltes: gescheitert", annotation_position="top left")
    fig.update_yaxes(title="Median-Fehler (Balken bis 90. Perzentil), %", range=[0, max(35.0, 100.0 * max(v["p90"] for v in d["rules"].values()) * 1.15)])
    return _fixed(fig, 380, showlegend=False)


def build_rounding_chart(res: dict) -> go.Figure:
    ks = res["k"][1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=res["exact"][1:], mode="lines+markers", name="exakt (volle Reorthogonalisierung)", line=dict(color=C.COLOR_CGLS, width=3), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=ks, y=res["plain"][1:], mode="lines+markers", name="übliche Rekurrenz, doppelte Genauigkeit", line=dict(color=C.COLOR_PLAIN, width=3, dash="dash"), marker=dict(size=4)))
    fig.update_xaxes(title="Iteration k")
    fig.update_yaxes(title="RMS-Fehler relativ zur Signalnorm", type="log", range=[-1.2, 0.8], **ERR_TICKS)
    return _fixed(fig, 380)
