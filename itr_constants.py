"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: SVD einer 96 x 96-Matrix (O(n^3), unter 20 ms); CGLS über höchstens 40 Schritte mit voller Reorthogonalisierung
# (O(k^2 n) je Wiederholung), die Experimente nutzen feste Ströme mit 200 Wiederholungen.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)        # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
DATA_CHOICES = ("ehrlich", "verbrechen")
DATA_LABELS = {"ehrlich": "Ehrliche Daten (analytisch, vom Gitter unabhängig)", "verbrechen": "Daten aus demselben Gitter (inverse crime)"}
METHODS = ("cgls", "landweber", "cgls_plain")
METHOD_LABELS = {"cgls": "CGLS, exakt (volle Reorthogonalisierung)", "landweber": "Landweber (Schrittweite 1/σ₁²)", "cgls_plain": "CGLS, übliche Rekurrenz (doppelte Genauigkeit)"}
METHOD_SHORT = {"cgls": "CGLS exakt", "landweber": "Landweber", "cgls_plain": "CGLS übliche Rekurrenz"}
K_MIN, K_MAX, K_STEP, DEFAULT_K = 0, 40, 1, 30                    # Regler: bei CGLS k Schritte, bei Landweber 10^(Regler/5) Schritte (0 bis 10^8)
LANDWEBER_DECADE = 5                                              # Regler-Einheiten je Zehnerpotenz bei Landweber
SEED_MAX, DEFAULT_SEED = 999999, 11

KMAX = 40                                                         # größter CGLS-Index: bis hierher stimmt die Reorthogonalisierung auf 1e-14 mit 100 Stellen überein
SINGULAR_COND = 1e15                                              # ab hier ist die Matrix in doppelter Genauigkeit numerisch singulär
SHOW_RANGE = (-1.5, 2.0)                                          # Wertebereich der Rekonstruktion im Diagramm; größere Werte werden abgeschnitten
FAIL_FACTOR = 2.0                                                 # „gescheitert“: Fehler über dem Doppelten des erwarteten Optimums
LAM_GRID = (1e-9, 10.0, 241)                                      # Gitter für das beste Tikhonov-λ (Vergleich)
LW_GRID = (0, 8, 321)                                             # Landweber: Zehnerlogarithmus der Iterationszahl von 1 bis 10^8

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 990000
EXP_REPS = 200
EXP_NOISE_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4)                       # Experiment 2
EXP_OMEGA_FACTORS = (0.25, 0.5, 1.0, 1.5, 1.9)                    # Experiment 2: Schrittweite als Vielfaches von 1/σ₁²
EXP_OFFSETS = (0.5, 0.7, 1.0, 1.4, 2.0)                           # Experiment 1: Vielfache des besten Abbruchs
EXP_PLAIN_KS = (20, 25, 28, 30, 33, 36, 40)                       # Experiment 4: Abbruchstellen im Vergleich exakt gegen übliche Rekurrenz

COLOR_TRUTH = "#2F6B65"
COLOR_NAIVE = "#c0392b"
COLOR_CGLS = "#4c78a8"
COLOR_LW = "#D68A2E"
COLOR_PLAIN = "#8e44ad"
COLOR_TIK = "#555555"
COLOR_GREY = "#8a8a8a"
METHOD_COLORS = {"cgls": COLOR_CGLS, "landweber": COLOR_LW, "cgls_plain": COLOR_PLAIN}

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "method": "cgls", "k": 30, "seed": DEFAULT_SEED},
    "Landweber": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "method": "landweber", "k": 20, "seed": DEFAULT_SEED},
    "Zu spät": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "method": "cgls", "k": 38, "seed": DEFAULT_SEED},
    "Rundung": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "method": "cgls_plain", "k": 40, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "CGLS nach 30 Schritten: nahe dem besten Abbruch, der Fehler liegt bei rund 12 %.",
    "Landweber": "Landweber nach 10 000 Schritten (Regler 20): derselbe Fehler wie CGLS nach 30, aber mit 20 000 statt 60 Matrix-Vektor-Produkten.",
    "Zu spät": "CGLS nach 38 Schritten: die Iteration beginnt, das Rauschen der kleinen Singulärwerte aufzunehmen; der Fehler dieser Stichprobe liegt bei rund 26 %.",
    "Rundung": "Die übliche CGLS-Rekurrenz in doppelter Genauigkeit nach 40 Schritten: der Fehler bleibt bei rund 12 %, obwohl die exakte Iteration bei rund 40 % läge. Die Rundung regularisiert mit.",
}
