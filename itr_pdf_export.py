"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import itr_constants as C
from itr_evaluation import Analysis, de, err_pct, int_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "τ": "tau", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0", "¹": "1", "²": "2",
            "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Iterative Regularisierung: Abbruch als Parameter"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    kind = "Daten aus demselben Gitter (inverse crime)" if s.data == "verbrechen" else "ehrliche Daten (analytisch)"
    pdf.multi_cell(0, 5, _s(f"Gitterpunkte n = {s.n}, Breite der Unschärfe w = {de(s.w, 3)}, Rauschen {de(100 * s.noise, 4)} % des Effektivwerts, {kind}, "
                            f"Verfahren: {C.METHOD_SHORT[s.method]}, Schritte {int_text(a.steps)}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Gewählt: {C.METHOD_SHORT[s.method]} nach {int_text(a.steps)} Schritten ({int_text(a.matvec)} Matrix-Vektor-Produkte), Fehler dieser Stichprobe {err_pct(a.err_iter)}.",
        f"Bester CGLS-Abbruch dieser Stichprobe (Wahrheit bekannt): k = {a.k_oracle}, Fehler {err_pct(a.err_oracle)}.",
        f"Tikhonov beim besten erwarteten Parameter: lambda = {sci(a.lam_star)}, erwarteter Fehler {err_pct(a.err_tik)}; direkte Inversion {err_pct(a.err_naive)}.",
        "Abbruchregeln auf CGLS (diese Stichprobe): " + "; ".join(f"{n} k = {a.picks[k]}, Fehler {err_pct(a.pick_errors[k])}"
                                                                for k, n in (("dp", "Diskrepanzprinzip"), ("gcv", "GCV"), ("lcurve", "L-Kurve"), ("oracle", "Orakel"))) + ".",
        f"Kondition der Matrix: {sci(a.cond)}; absolutes Rauschen sigma = {sci(a.sigma)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Gewählte Iterierte", "Bester CGLS", "Tikhonov"], [16, 26, 30, 40, 38, 32]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.x_iter[i], 3), de(a.x_exact_best[i], 3), de(a.x_tik[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Landweber und CGLS mit Abbruch als Regularisierungsparameter. Quelle: iterativ-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
