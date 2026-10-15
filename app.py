"""Iterative Regularisierung: Landweber, CGLS und der Abbruch als Parameter - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 4 (Phantom, Gauß-Unschärfe, Rauschen) regularisiert hier nicht eine
Strafe, sondern die Zahl der Iterationen: Landweber (Gradientenabstieg auf dem Residuum) und CGLS (konjugierte Gradienten auf den Normalgleichungen). Kern:
itr_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import itr_constants as C
from itr_evaluation import (
    Settings,
    analyse,
    cost_experiment,
    de,
    err_pct,
    halfconv_experiment,
    int_text,
    landweber_steps,
    noise_label,
    rounding_experiment,
    sci,
    share,
    stop_experiment,
    verdict_cost,
    verdict_halfconv,
    verdict_instance,
    verdict_rounding,
    verdict_stop,
)
from itr_pdf_export import build_pdf
from itr_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from itr_visualization import (
    RULE_SHORT,
    build_cost_chart,
    build_halfconv_chart,
    build_landweber_chart,
    build_recon_chart,
    build_rounding_chart,
    build_semiconv_chart,
    build_stop_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Iterative Regularisierung – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _halfconv():
    return halfconv_experiment()


@st.cache_data(show_spinner=SPINNER)
def _cost():
    return cost_experiment()


@st.cache_data(show_spinner=SPINNER)
def _stop():
    return stop_experiment()


@st.cache_data(show_spinner=SPINNER)
def _rounding():
    return rounding_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🔁 Iterative Regularisierung: Abbruch als Parameter")
st.markdown(
    """
**Hier regularisiert die Zahl der Schritte.** Dies ist das fünfte Stück der Linie **Inverse Probleme und Regularisierung** und baut auf **Tikhonov**, **TSVD** und der
**Parameterwahl** auf: dasselbe Phantom, dieselbe Gauß-Unschärfe, dasselbe Rauschen. **Landweber** (ein Gradientenschritt auf dem Residuum) und **CGLS** (konjugierte Gradienten
auf den Normalgleichungen) starten bei null und nähern sich der direkten Inversion; **zu früh gestoppt ist die Lösung zu glatt, zu spät ist sie vom Rauschen verdorben**. Der
Abbruch ist der Regularisierungsparameter. Die Welt ist simuliert; der beste Abbruch dieser Stichprobe und Tikhonov mit dem besten λ stehen als Maßstab daneben.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie ein Iterationsverfahren regularisiert", expanded=True):
    st.markdown(
        r"""
1. **Landweber.** $x_{k+1} = x_k + \omega A^\top (b - A x_k)$ mit $\omega = 1/\sigma_1^2$ und $x_0 = 0$. In den Singulärkoordinaten ist das ein Filter: Die Komponente mit Singulärwert
   $\sigma_i$ ist nach $k$ Schritten zu $1 - (1 - \omega\sigma_i^2)^k$ aufgebaut. Große Singulärwerte kommen zuerst, kleine erst nach sehr vielen Schritten.
2. **CGLS.** Wählt in jedem Schritt die beste Lösung im Krylov-Raum $\mathrm{span}\{A^\top b, (A^\top A)A^\top b, \dots\}$; das braucht dieselben zwei Matrix-Vektor-Produkte je Schritt,
   aber weit weniger Schritte.
3. **Abbruch.** Beide Verfahren konvergieren gegen die direkte Inversion, also gegen die Lösung mit dem Rauschen. Dazwischen liegt ein Minimum des Fehlers: *Halbkonvergenz*.
   Wo man anhält, bestimmen die Regeln aus Stück 4, jetzt auf dem Iterationsindex.
4. **Maßstab.** Der *beste CGLS-Abbruch dieser Stichprobe* braucht die Wahrheit; Tikhonov mit dem besten erwarteten λ ist die Referenz aus Stück 2.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Messung**")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Messpunkte und Unbekannte.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    st.markdown("**Daten**")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_CHOICES, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrliche Daten kommen aus der analytischen Unschärfe und sind vom Gitter unabhängig. Daten aus demselben Gitter entstehen aus "
                             "der Matrix, die danach invertiert wird (inverse crime).")
    st.markdown("**Verfahren**")
    method = st.selectbox("Welches Verfahren?", C.METHODS, key="method_select", format_func=lambda v: C.METHOD_LABELS[v],
                          help="CGLS exakt rechnet mit voller Reorthogonalisierung und zeigt die Halbkonvergenz, die in exakter Arithmetik gilt. Die übliche Rekurrenz in "
                               "doppelter Genauigkeit stagniert bei großem k, weil die Rundung mitregularisiert.")
    k = st.slider("Abbruch (Regler)", *bounds("k_slider"), step=C.K_STEP, key="k_slider",
                  help="Bei CGLS die Zahl der Schritte k. Bei Landweber zählt der Regler in Zehnerpotenzen: Schritte = 10^(Regler/5), also 0 bis 10^8.")
    shown_steps = landweber_steps(int(k)) if method == "landweber" else int(k)
    st.caption(f"= {int_text(shown_steps)} Schritte, {int_text(2 * shown_steps)} Matrix-Vektor-Produkte")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Rauschen würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "data_select": data, "method_select": method,
                   "k_slider": int(k), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), data, method, int(k), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.singular:
    st.warning(f"Bei dieser Wahl von n und w ist die Matrix in doppelter Genauigkeit numerisch singulär (Kondition {sci(a.cond)}): Die kleinen Singulärwerte sind Rundungsrauschen, "
               "und die Zahlen unten sind nur noch Zahlen.")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{C.METHOD_SHORT[method]}: Fehler", err_pct(a.err_iter), delta=f"{int_text(a.steps)} Schritte, {int_text(a.matvec)} Matrix-Vektor-Produkte", delta_color="off",
          help="‖x̂ − x‖ / ‖x‖ mit der Wahrheit x des Phantoms.")
c2.metric("Bester CGLS-Abbruch dieser Stichprobe", err_pct(a.err_oracle), delta=f"k = {a.k_oracle}", delta_color="off",
          help="Der Index mit dem kleinsten tatsächlichen Fehler dieser Stichprobe; braucht die Wahrheit.")
c3.metric("Tikhonov, bestes λ", err_pct(a.err_tik), delta=f"λ* = {sci(a.lam_star)} (erwartet)", delta_color="off",
          help="Erwarteter Fehler bei dem λ, das den erwarteten Fehler minimiert (Stück 2).")
c4.metric("Direkte Inversion", err_pct(a.err_naive), delta="unendlich viele Schritte", delta_color="off", help="Fehler dieser Stichprobe ohne jede Regularisierung.")
if method == "landweber" and a.steps > 0:
    st.caption(f"Landweber nach {int_text(a.steps)} Schritten wirkt wie Tikhonov mit λ ≈ 1/√(ωk) = {sci(a.lam_equiv)} (nur als Größenordnung: die Filter haben verschiedene Form).")

tab_rec, tab_semi, tab_lw, tab_tab = st.tabs(["Rekonstruktion", "Halbkonvergenz (CGLS)", "Landweber über Schritte", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, grau gestrichelt der beste CGLS-Abbruch dieser Stichprobe, punktiert Tikhonov, farbig die gewählte Iterierte, blass rot die direkte Inversion. "
               "Die Achse ist fest.")
with tab_semi:
    st.plotly_chart(build_semiconv_chart(a), width="stretch", key="semiconv_chart")
    st.caption("Der Fehler dieser Stichprobe über den CGLS-Index. Die exakte Iteration (blau) steigt nach dem Minimum steil; die übliche Rekurrenz in doppelter Genauigkeit "
               "(violett gestrichelt) bleibt unten, weil die Rundung mitregularisiert.")
with tab_lw:
    st.plotly_chart(build_landweber_chart(a), width="stretch", key="landweber_chart")
    st.caption("Der erwartete Fehler von Landweber (geschlossen, über das Rauschen gemittelt) über die Schrittzahl auf logarithmischer Achse: das Minimum liegt nach Tausenden von Schritten.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="iterativ.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann hört man auf?")
r1, r2, r3 = st.columns(3)
for col, key in zip((r1, r2, r3), ("dp", "gcv", "lcurve")):
    col.metric(f"{RULE_SHORT[key]}: Fehler dieser Stichprobe", err_pct(a.pick_errors[key]), delta=f"stoppt bei k = {a.picks[key]}", delta_color="off")
verdict(verdict_instance(a))
st.caption("Eine einzelne Stichprobe zeigt nicht, wie oft eine Regel scheitert: dafür sind die Experimente mit festen Wiederholungen da.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Halbkonvergenz und Nachsicht")
res1 = _halfconv()
st.caption(f"Standardinstanz (n = {res1['n']}), ehrliche Daten, {res1['reps']} feste Wiederholungen: der RMS-Fehler der exakten CGLS-Iteration über k, mit den Referenzen.")
st.plotly_chart(build_halfconv_chart(res1), width="stretch", key="halfconv_chart")
verdict(verdict_halfconv(res1))
st.dataframe(
    [{"Vielfaches des besten Abbruchs": de(o["factor"], 1), "CGLS k": o["k"], "CGLS RMS-Fehler": err_pct(o["rms"]), "Landweber Schritte": int_text(lw["k"]),
      "Landweber erwarteter Fehler": err_pct(lw["err"])} for o, lw in zip(res1["offsets"], res1["lw_offsets"])],
    hide_index=True, width="stretch")
st.caption(f"Ein Vielfaches über 1 wird bei CGLS auf k = {C.KMAX} gekappt. Landweber verzeiht den Abbruch viel mehr: Sein Filter wächst langsam in die kleinen Singulärwerte hinein, "
           "CGLS springt mit den Ritz-Werten.")

st.subheader("🔬 Experiment 2: Was es kostet")
res2 = _cost()
st.caption(f"Standardinstanz (n = {res2['n']}), ehrliche Daten, {res2['reps']} feste Wiederholungen je Rauschstufe: Matrix-Vektor-Produkte bis zum besten Abbruch.")
st.plotly_chart(build_cost_chart(res2), width="stretch", key="cost_chart")
verdict(verdict_cost(res2))
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "CGLS bester k": r["cgls_k"], "CGLS Fehler (RMS)": err_pct(r["cgls_err"]), "Landweber beste Schritte": int_text(r["lw_k"]),
      "Landweber Fehler": err_pct(r["lw_err"]), "Tikhonov Fehler": err_pct(r["tik_err"])} for r in res2["rows"]],
    hide_index=True, width="stretch")
st.markdown("**Schrittweite** (Vielfaches von 1/σ₁², Standardinstanz, bester Abbruch)")
st.dataframe([{"ω · σ₁²": de(o["factor"], 2), "beste Schritte": int_text(o["k"]), "erwarteter Fehler": err_pct(o["err"])} for o in res2["omegas"]], hide_index=True, width="stretch")
st.caption(f"Die Schrittweite verschiebt nur den Schrittzähler: Der beste Fehler bleibt gleich, die Zahl der Schritte skaliert mit 1/ω. Über 2/σ₁² ist Landweber instabil: "
           f"mit 2,1/σ₁² liegt der erwartete Fehler nach 50 Schritten beim {sci(res2['unstable'])}-Fachen der Signalnorm.")

st.subheader("🔬 Experiment 3: Wann stoppen?")
res3 = _stop()
st.caption(f"Standardinstanz (n = {res3['n']}), {res3['reps']} feste Wiederholungen je Datenart: die Regeln aus Stück 4 auf dem CGLS-Index; "
           "gescheitert heißt: mehr als das Doppelte des erwarteten Optimums.")
kind = st.radio("Datenart", ("honest", "crime"), key="stop_kind", horizontal=True, format_func=lambda v: "ehrliche Daten" if v == "honest" else "Daten aus demselben Gitter")
st.plotly_chart(build_stop_chart(res3, kind), width="stretch", key="stop_chart")
verdict(verdict_stop(res3))
d3 = res3["kinds"][kind]
st.dataframe(
    [{"Regel": RULE_SHORT[k_], "Median k": int(v["k_median"]), "5 % / 95 % von k": f"{int(v['k_q'][0])} / {int(v['k_q'][1])}", "Median": err_pct(v["median"]),
      "90. Perzentil": err_pct(v["p90"]), "RMS": err_pct(v["rms"]), "Gescheitert": share(v["fail"])} for k_, v in d3["rules"].items()],
    hide_index=True, width="stretch")
ch, cc = res3["kinds"]["honest"], res3["kinds"]["crime"]
st.caption(f"Bester k im Mittel: {d3['k_star']} mit {err_pct(d3['err_star'])}. Das Diskrepanzprinzip stoppt dicht davor und bleibt flach. Die L-Kurve nimmt als Ecke die Stelle vor der Klippe "
           f"(Median k = {int(d3['rules']['lcurve']['k_median'])}) und scheitert dort bei {share(ch['rules']['lcurve']['fail'])} der ehrlichen Daten und {share(cc['rules']['lcurve']['fail'])} "
           f"der Daten aus demselben Gitter. Bei diesen ist das Optimum weit besser erreichbar ({err_pct(cc['err_star'])} gegen {err_pct(ch['err_star'])}), und GCV scheitert häufiger "
           f"({share(cc['rules']['gcv']['fail'])} gegen {share(ch['rules']['gcv']['fail'])}).")

st.subheader("🔬 Experiment 4: Die Rundung lügt")
res4 = _rounding()
st.caption(f"Standardinstanz (n = {res4['n']}), ehrliche Daten, {res4['reps']} feste Wiederholungen: dieselben Daten, zwei Rechenwege.")
st.plotly_chart(build_rounding_chart(res4), width="stretch", key="rounding_chart")
verdict(verdict_rounding(res4))
st.dataframe([{"k": r["k"], "exakt (RMS)": err_pct(r["exact"]), "übliche Rekurrenz (RMS)": err_pct(r["plain"])} for r in res4["rows"]], hide_index=True, width="stretch")
st.caption("Die übliche Rekurrenz verliert die Orthogonalität der Suchrichtungen; in doppelter Genauigkeit stagniert sie bei großem k und sieht gutmütiger aus, als die Iteration "
           "ist. Die Demo rechnet deshalb die exakte Iteration (volle Reorthogonalisierung) und zeigt die übliche daneben.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
r_mid = next(r for r in res2["rows"] if abs(r["noise"] - C.DEFAULT_NOISE) < 1e-12)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Exakte Arithmetik bis k = {C.KMAX}** | Die Iteration ist bis k = {C.KMAX} gegen 100 Dezimalstellen geprüft; darüber würde die volle Reorthogonalisierung selbst an die Rundung stoßen. Die übliche Rekurrenz in doppelter Genauigkeit liegt bei k = {C.KMAX} bei {err_pct(res4['rows'][-1]['plain'])} statt {err_pct(res4['rows'][-1]['exact'])}. | – |
| **Schrittweite 1/σ₁² bei Landweber** | Die Schrittweite verschiebt nur den Zähler (gleicher bester Fehler); jenseits von 2/σ₁² divergiert das Verfahren. | – |
| **Der Abbruch als einziger Parameter** | Landweber und CGLS ohne Strafterm: kein Vorwissen über die Lösung geht ein. Mit Vorwissen (Glätte, Sparsamkeit) lässt sich das Minimum von {err_pct(r_mid['tik_err'])} unterschreiten. | Sparsity und TV (Stück 6) |
| **GCV mit df = k** | Für CGLS ist k nur eine Näherung der Freiheitsgrade; die Katastrophenrate der Regel ist hier gemessen, nicht hergeleitet. | – |
| **Das Rauschen ist weiß und Gauß-verteilt** | Alle Regeln und das Diskrepanzprinzip nehmen weißes Rauschen an; korreliertes Rauschen verschiebt den besten Abbruch. | – |
| **Ein Phantom, eine Matrix** | Eine Welt, ein Rauschen; Stoppregeln und Vergleich gelten für dieses Beispiel. | – |
| **Das Orakel kennt die Wahrheit** | Der beste Abbruch und das beste λ sind in der Praxis nicht verfügbar; sie sind nur der Maßstab. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Landweber.** $x_{k+1} = x_k + \omega A^\top (b - A x_k)$, $x_0 = 0$. Mit $A = U\,\mathrm{diag}(\sigma_i)\,V^\top$ und $\beta = U^\top b$ ist $x_k = \sum_i f_i\,(\beta_i/\sigma_i)\,v_i$
mit $f_i = 1 - (1 - \omega\sigma_i^2)^k$ für $0 < \omega < 2/\sigma_1^2$. Der erwartete Fehler folgt aus den Filterfaktoren:
$\sqrt{\sum_i [f_i(c_i + \rho_i) - c_i]^2 + f_i^2\sigma^2/\sigma_i^2}\,/\,\|x\|$ mit $c = V^\top x$ und dem Modellfehler $\rho_i = (U^\top r)_i/\sigma_i$.

**CGLS.** $x_k = \arg\min\{\|b - Ax\| : x \in \mathcal{K}_k(A^\top A, A^\top b)\}$. Die Iterierten stammen aus der Golub-Kahan-Bidiagonalisierung $A V_k = U_{k+1} B_k$;
$x_k = V_k z_k$ mit $z_k = \arg\min\|B_k z - \|b\|\,e_1\|$. In der Demo wird $V_k$ voll reorthogonalisiert und gegen CGLS in 100 Dezimalstellen geprüft.

**Halbkonvergenz.** Der Fehler $\|x_k - x\|$ fällt, solange die Iteration die Komponenten mit $|c_i| \gg \sigma/\sigma_i$ aufbaut, und steigt, sobald sie Komponenten mit $\sigma_i < \sigma/|c_i|$
aufnimmt: dort wird das Rauschen verstärkt. Das Minimum ist der Abbruch als Regularisierungsparameter.

**Abbruchregeln.** Diskrepanzprinzip: erstes $k$ mit $\|Ax_k - b\|^2 \le \sigma^2 n$. GCV: Minimum von $\|Ax_k - b\|^2/(n - k)^2$ (df = k). L-Kurve: größte Krümmung von
$(\log\|Ax_k - b\|, \log\|x_k\|)$ über $k$.

**Literatur.** Das Beispiel ist erfunden; Landweber (1951), CGLS und die Krylov-Räume (Hestenes und Stiefel 1952) und die Halbkonvergenz sind bei P. C. Hansen,
*Discrete Inverse Problems: Insight and Algorithms*, SIAM 2010, und M. Hanke, *Conjugate Gradient Type Methods for Ill-Posed Problems*, 1995, beschrieben (aus dem Gedächtnis,
nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
