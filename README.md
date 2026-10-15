# Iterative Regularisierung: Landweber, CGLS und der Abbruch als Parameter – Streamlit-Demo

Fünftes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo), [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo), [tsvd-demo](https://github.com/sebastian-hanisch/tsvd-demo) und [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab, nie als Eingabe eines Verfahrens.

## Kernfrage

Bisher regularisierte eine Strafe (Tikhonov) oder ein Rang (TSVD). Hier regularisiert die **Zahl der Iterationen**: Landweber und CGLS starten bei null und nähern sich der direkten Inversion, der Abbruch ist der Parameter. Wie sieht die Halbkonvergenz aus, was kostet sie an Matrix-Vektor-Produkten, wo hört man auf, und was verdeckt die Rundung?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, analytische Unschärfe, Gitter, Rauschen, zwei Datenarten):

- **Wahrheit** x(s) auf [0, 1]: zwei Gauß-Hügel und ein Kasten; **Messung** mit dem Gauß-Kern der Breite w, analytisch bekannt.
- **Gitter** mit n Punkten. Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 3,4·10⁻⁴), Kondition 1,26·10⁴; die direkte Inversion liegt bei 237 % Fehler.
- **Daten:** *ehrlich* (analytisch, vom Gitter unabhängig) oder *aus demselben Gitter* (b = A x + e, „inverse crime“); das Rauschen ist in beiden gleich.
- **Landweber** x₍ₖ₊₁₎ = xₖ + ω Aᵀ(b − A xₖ) mit ω = 1/σ₁² und x₀ = 0; der Regler zählt in Zehnerpotenzen (0 bis 10⁸ Schritte). Der Filter ist geschlossen 1 − (1 − ωσᵢ²)ᵏ, deshalb sind auch zehntausend Schritte exakt und billig.
- **CGLS** (konjugierte Gradienten auf den Normalgleichungen) wird in zwei Fassungen gerechnet: *exakt* über die Golub-Kahan-Bidiagonalisierung mit voller Reorthogonalisierung (bis k = 40, gegen 100 Dezimalstellen geprüft) und in der *üblichen Rekurrenz* in doppelter Genauigkeit.

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Abbruchregeln auf dem Iterationsindex** (CGLS exakt): Diskrepanzprinzip (erstes k mit ‖A xₖ − b‖² ≤ σ²n), GCV mit df = k (eine Näherung), L-Kurve (größte Krümmung von (log ‖A xₖ − b‖, log ‖xₖ‖) über k).
- **Maßstab:** Tikhonov beim besten erwarteten λ (aus Stück 2), der beste CGLS-Abbruch der einzelnen Stichprobe (Realisierungsorakel) und der beste erwartete Abbruch über die Wiederholungen. „Gescheitert“ heißt: Fehler über dem Doppelten des erwarteten Optimums.
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 200 Wiederholungen je Zelle, nie ein einzelner Seed. Die Primäransicht ist eine einzelne Stichprobe und sagt nichts über die Häufigkeit des Scheiterns.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03, Rauschen 0,1 %, ehrliche Daten), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Wie sieht die Halbkonvergenz aus (Experiment 1)? | CGLS, RMS-Fehler: k = 1: 30,7 %; 3: 15,8 %; 5: 14,7 %; 10: 14,2 %; 20: 12,9 %; **Minimum bei k = 30 mit 11,6 %**; 33: 13,1 %; 35: 15,9 %; 38: 26,7 %; **40: 45,1 %**. Die Strafe für zu spät ist steil, die für zu früh flach: das 0,7-fache des besten k kostet 12,7 %, das 1,4-fache (auf 40 gekappt) 45,1 %. | `test_semi_convergence_numbers`, `test_references_and_landweber_forgiveness` |
| Gegen TSVD und Tikhonov? | Tikhonov beim besten λ: 11,7 %; TSVD (Rang 33): 11,4 %; CGLS beim besten k: 11,6 %. Die drei Wege führen zum selben Fehler. | `test_references_and_landweber_forgiveness` |
| Landweber: gleich gut, anders? | Bestes Minimum bei **8.913 Schritten** mit 11,6 % (17.826 Matrix-Vektor-Produkte gegen 60 bei CGLS). Von 0,5- bis 2,0-fach des besten Abbruchs bleibt der erwartete Fehler **zwischen 11,6 und 11,7 %**: Landweber verzeiht den Abbruch, CGLS nicht. | `test_references_and_landweber_forgiveness`, `test_cost_numbers` |
| Was kostet es über Rauschstufen (Experiment 2)? | Bester Abbruch für Rauschen 10 % / 1 % / 0,1 % / 0,01 %: **CGLS 3 / 19 / 30 / 32 Schritte** (17,6 / 13,6 / 11,5 / 11,3 %), **Landweber 7 / 316 / 8.913 / 19.953 Schritte** (17,6 / 13,4 / 11,6 / 11,3 %). Das Verhältnis der Matrix-Vektor-Produkte steigt von 2 auf 624. | `test_cost_numbers` |
| Was macht die Schrittweite? | ω·σ₁² = 0,25 / 0,5 / 1 / 1,5 / 1,9: beste Schritte 35.481 / 17.783 / 8.913 / 5.957 / 4.732, **derselbe Fehler** (11,59 %); die Schrittweite skaliert nur den Zähler. Mit 2,1/σ₁² divergiert das Verfahren: nach 50 Schritten das 83-Fache der Signalnorm. | `test_step_size_changes_only_the_step_count` |
| Wann stoppen (Experiment 3, ehrliche Daten)? | Bester k im Mittel 30 (11,6 %). **Diskrepanzprinzip:** Median k = 24, Fehler 12,0 %, 90. Perzentil 12,2 %, **0 %** gescheitert. **GCV (df = k):** Median k = 29, Fehler 11,8 %, aber **12,5 % gescheitert**, 90. Perzentil 28,7 %. **L-Kurve:** Median k = 37, Fehler 19,2 %, **26,5 %** gescheitert. Orakel 11,5 %. | `test_stop_rule_numbers_honest` |
| Und mit Daten aus demselben Gitter? | Optimum bei k = 31 mit **7,1 %**. Diskrepanzprinzip 8,3 % (0 % gescheitert), GCV Median 7,7 % mit **30 %** gescheitert (90. Perzentil 40 %), L-Kurve Median 15,5 % mit **56,5 %** gescheitert. Orakel 7,0 %. | `test_stop_rule_numbers_crime` |
| Was verdeckt die Rundung (Experiment 4)? | Gleiche Daten, zwei Rechenwege, RMS-Fehler bei k = 20 / 28 / 30 / 33 / 36 / 40: **exakt** 12,9 / 11,6 / 11,6 / 13,1 / 17,9 / **45,1 %**, **übliche Rekurrenz** 12,9 / 11,9 / 11,8 / 11,7 / 11,6 / **11,6 %**. In doppelter Genauigkeit sieht CGLS gutmütiger aus, als die Iteration ist. | `test_rounding_numbers` |
| Einzelstichproben (Presets) | Standard (Seed 11, CGLS k = 30): 11,8 %, bester Abbruch dieser Stichprobe k = 28 mit 11,6 %. Landweber nach 10.000 Schritten: 11,8 % mit 20.000 statt 60 Matrix-Vektor-Produkten. Zu spät (k = 38): **25,7 %**. Rundung (übliche Rekurrenz, k = 40): 11,7 % statt 39,7 % der exakten Iteration. | `test_standard_instance_numbers`, `test_preset_help_numbers` |
| Mini-Instanz von Hand | 3 × 3-Matrix (Singulärwerte 1 / 0,5 / 0,1), rauschfreie Daten: Landweber-Filter 1 / 0,25 / 0,01 (k = 1), 1 / 0,4375 / 0,0199 (k = 2); erster Schritt Aᵀb = (1,33; 1,34; 1,33); CGLS im ersten Schritt 1,0000125·Aᵀb und nach zwei Schritten exakt (nur zwei Komponenten tragen Daten). | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Die L-Kurve wählte zunächst die flachste statt der schärfsten Stelle.** Die Vormessung hatte „L-Kurve wählt k = 3 und scheitert nie“ ergeben. Der von-Hand-Test an einer gebauten Ecke fand das Vorzeichen: Die Kurve wird mit wachsendem k in umgekehrter Richtung durchlaufen wie bei wachsendem λ, der Zähler der Krümmung muss das umgekehrte Vorzeichen tragen. Richtig gerechnet nimmt die Regel k = 37 kurz vor der Klippe und scheitert bei 27 % der ehrlichen und 57 % der Daten aus demselben Gitter, wie im Stück 4 bei Tikhonov.
- **GCV scheitert hier bei 12,5 %** (200 feste Wiederholungen), nicht bei den 14 % der Vormessung (400); die Aussage „das Diskrepanzprinzip trifft, GCV scheitert in jedem achten Fall“ trägt.
- **Die Rundung regularisiert mit.** Die Vormessung in doppelter Genauigkeit zeigte keine Halbkonvergenz bis k = 40; erst die Reorthogonalisierung (gegen `decimal` in 100 Stellen geprüft) zeigt sie. Die Demo rechnet deshalb beide Wege und weist darauf hin.
- **CGLS ist schon nach drei Schritten bei 15,8 %.** Der Gewinn der letzten 27 Schritte (auf 11,6 %) ist klein gegen den Verlust beim Überschreiten.
- **Landweber ist schrittweitenunabhängig im Fehler**, nicht im Aufwand: Schrittweite und Schrittzahl sind austauschbar.
- **Gescheitert ist eine Konvention** (mehr als das Doppelte des erwarteten Optimums); die Anteile ändern sich mit ihr, die Rangfolge der Regeln nicht.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Exakte Arithmetik bis k = 40** | Darüber würde auch die volle Reorthogonalisierung an die Rundung stoßen; die übliche Rekurrenz liegt bei k = 40 bei 11,6 % statt 45,1 %. | – |
| **Schrittweite 1/σ₁² bei Landweber** | Die Schrittweite verschiebt nur den Zähler; über 2/σ₁² divergiert das Verfahren. | – |
| **Der Abbruch als einziger Parameter** | Kein Vorwissen über die Lösung geht ein; mit Vorwissen (Glätte, Sparsamkeit) lässt sich das Minimum von rund 11,6 % unterschreiten. | Sparsity und TV (Stück 6) |
| **GCV mit df = k** | Für CGLS ist k nur eine Näherung der Freiheitsgrade; die Katastrophenrate ist gemessen, nicht hergeleitet. | – |
| **Das Rauschen ist weiß und Gauß-verteilt** | Alle Regeln nehmen weißes Rauschen an. | – |
| **Ein Phantom, eine Matrix** | Die Zahlen gelten für dieses Beispiel. | – |
| **Das Orakel kennt die Wahrheit** | Der beste Abbruch und das beste λ sind in der Praxis nicht verfügbar. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo): Stück 4; dieselben drei Regeln, dort auf λ, hier auf dem Iterationsindex.
- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und [tsvd-demo](https://github.com/sebastian-hanisch/tsvd-demo): Stück 2 und 3; die Vergleichswerte 11,7 % und 11,4 %.

## Tests

Mini-Instanz von Hand (3 × 3: Filter, erster Schritt, Abbruch, erschöpfter Krylov-Raum), Landweber gegen die explizite Iteration, CGLS gegen eine Normalgleichungs-CG in x-Koordinaten, gegen die Krylov-Definition (kleinste Quadrate über den Krylov-Raum) und gegen CGLS in 100 Dezimalstellen bis k = 40, Erwartung gegen Monte Carlo, Abbruchregeln gegen Schleifen-Implementierungen, Generator (SplitMix64 gegen eine unabhängige Fassung), Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Slider-Grenzen, jede Rauschstufe, jedes Verfahren und jede Datenart, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `itr_model.py` | Landweber-Filter, CGLS (exakt und übliche Rekurrenz), Abbruchregeln, erwarteter Fehler |
| `itr_evaluation.py` | Kennzahlen, Experimente, Verdikte |
| `itr_visualization.py`, `itr_pdf_export.py` | Figuren, PDF |
| `itr_constants.py`, `itr_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Weitere Verfahren** (LSQR mit Dämpfung, MR-II, Arnoldi-Tikhonov, Hybridverfahren): ändern die Aussage nicht.
- **Vorkonditionierung und Glättungsnormen** (L ≠ I): gehören zu einem späteren Stück.
- **Nichtlineare Iterationen** (Gauß-Newton, Levenberg-Marquardt): Stück 9.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
