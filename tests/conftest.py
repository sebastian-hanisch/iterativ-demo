"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 4 (symmetrisch, Zeilensumme 1, Singulärwerte 1, 0,5, 0,1; Eigenvektoren (1,1,1), (1,0,-1), (1,-2,1)), Wahrheit x = (1, 2, 1),
    rauschfreie Daten b = A x = (1,3, 1,4, 1,3), Komponenten beta = U^T b = (+-2,3094, 0, -+0,08165) (Vorzeichen der SVD beliebig).
    Handrechnung Landweber mit omega = 1: Filter f_i = 1 - (1 - sigma_i^2)^k mit sigma^2 = (1, 0,25, 0,01), also k = 1: (1, 0,25, 0,01), k = 2: (1, 0,4375, 0,0199),
    k = 3: (1, 0,578125, 0,029701); der erste Schritt ist A^T b = (1,33, 1,34, 1,33). CGLS: 3 verschiedene Singulärwerte, also ist der dritte Schritt exakt x;
    der erste ist alpha A^T b mit alpha = |A^T b|^2 / |A A^T b|^2 = 5,3334 / 5,333334 = 1,0000125."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b
