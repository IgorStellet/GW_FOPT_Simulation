"""LISA no plano de espectros: curva integrada em leis de potência (PLIS).

O PIS acústico continua sendo usado na classificação dos pontos. Esta curva
é uma referência gráfica para espectros completos, construída a partir do
ruído de Robson, Cornish & Liu, arXiv:1803.01944, eqs. (1), (10), (11), (14).
O envelope de leis de potência segue a definição discutida em arXiv:2002.04615.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import trapezoid

YEAR_SECONDS = 365.25 * 24 * 3600
H100_PER_SECOND = 100_000 / 3.085677581491367e22
CONFUSION = {
    0.5: (0.133, 243, 482, 917, 0.00258),
    1.0: (0.171, 292, 1020, 1680, 0.00215),
    2.0: (0.165, 299, 611, 1340, 0.00173),
    4.0: (0.138, -221, 521, 1680, 0.00113),
}


def lisa_noise_h2(f_Hz, *, foreground_years=4.0):
    """h² Omega_noise; a resposta já soma os dois canais do ajuste de ruído."""
    f = np.asarray(f_Hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Frequências LISA devem ser positivas e finitas, em Hz.")
    arm = 2.5e9
    fstar = 299_792_458 / (2 * math.pi * arm)
    optical = (1.5e-11) ** 2 * (1 + (2e-3 / f) ** 4)
    acceleration = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    strain = (
        10
        / (3 * arm**2)
        * (optical + 4 * acceleration / (2 * math.pi * f) ** 4)
        * (1 + 0.6 * (f / fstar) ** 2)
    )
    if foreground_years is not None:
        if foreground_years not in CONFUSION:
            raise ValueError(
                "Ajustes do foreground disponíveis para 0.5, 1, 2 e 4 anos."
            )
        a, b, k, g, fk = CONFUSION[foreground_years]
        strain += (
            9e-45
            * f ** (-7 / 3)
            * np.exp(-(f**a) + b * f * np.sin(k * f))
            * (1 + np.tanh(g * (fk - f)))
        )
    return 2 * math.pi**2 / (3 * H100_PER_SECOND**2) * f**3 * strain


@lru_cache(maxsize=16)
def _plis_amplitudes(years, threshold, foreground_years):
    if not all(math.isfinite(x) and x > 0 for x in (years, threshold)):
        raise ValueError("Tempo e limiar LISA devem ser positivos e finitos.")
    frequency = np.geomspace(1e-5, 1, 6000)
    noise = lisa_noise_h2(frequency, foreground_years=foreground_years)
    slopes = np.linspace(-8, 8, 321)
    shapes = (frequency[None, :] / 1e-3) ** slopes[:, None]
    integral = trapezoid((shapes / noise) ** 2, frequency, axis=1)
    amplitudes = threshold / np.sqrt(years * YEAR_SECONDS * integral)
    return slopes, amplitudes


def lisa_plis_h2(f_Hz, *, years=4.0, threshold=10.0, foreground_years=4.0):
    """Envelope PLIS em h²Omega, para a comparação visual de espectros."""
    f = np.asarray(f_Hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Frequências devem ser positivas e finitas.")
    slopes, amplitudes = _plis_amplitudes(years, threshold, foreground_years)
    return np.max(
        amplitudes[:, None] * (f.ravel()[None, :] / 1e-3) ** slopes[:, None], axis=0
    ).reshape(f.shape)
