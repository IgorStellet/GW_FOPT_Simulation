"""Sensibilidades: PIS acústico para seleção, PLIS LISA para mostrar espectros.

PIS: Schmitz, arXiv:2002.04615, normalização original 1 ano e SNR=1.
Ruído/PLIS: Robson, Cornish e Liu, arXiv:1803.01944, com foreground de 4 anos.
Todas as interfaces deste arquivo usam Hz; os ajustes PIS do src usam mHz.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import trapezoid

from .core import gw

YEAR_SECONDS = 365.25 * 86400
H100 = 100_000 / 3.085677581491367e22
PIS = {
    "LISA": gw.lisa_sensitivity_s_pis,
    "DECIGO": gw.decigo_sensitivity_s_pis,
    "BBO": gw.bbo_sensitivity_s_pis,
}


def pis_h2(frequency_Hz, detector="LISA", *, years=4.0, threshold=10.0):
    """Amplitude acústica de pico necessária para o limiar declarado."""
    f = np.asarray(frequency_Hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0) or years <= 0 or threshold <= 0:
        raise ValueError("Frequência, tempo e limiar devem ser positivos.")
    return PIS[detector](1e3 * f) * threshold / math.sqrt(years)


def acoustic_snr(row, detector="LISA", *, years=4.0):
    """SNR do canal acústico, sob a forma espectral do src usada pelos ajustes PIS."""
    if years <= 0:
        raise ValueError("Tempo de observação deve ser positivo.")
    return float(
        row["omega_sw_peak_h2"] / pis_h2(row["f_sw_peak_Hz"], detector, years=years, threshold=1)
    )


def lisa_noise_h2(frequency_Hz):
    """Ruído efetivo em h²Omega, dois canais, braços de 2.5 Gm e foreground de 4 anos."""
    f = np.asarray(frequency_Hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Frequências devem ser positivas e finitas.")
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
    strain += (
        9e-45
        * f ** (-7 / 3)
        * np.exp(-(f**0.138) - 221 * f * np.sin(521 * f))
        * (1 + np.tanh(1680 * (0.00113 - f)))
    )
    return 2 * math.pi**2 / (3 * H100**2) * f**3 * strain


@lru_cache(maxsize=8)
def _plis_coefficients(years, threshold):
    """Integra uma vez cada lei de potência, formando o envelope de sensibilidade."""
    if not math.isfinite(years) or not math.isfinite(threshold) or years <= 0 or threshold <= 0:
        raise ValueError("Tempo e limiar devem ser positivos e finitos.")
    f = np.geomspace(1e-5, 1, 6000)
    slopes = np.linspace(-8, 8, 321)
    shapes = (f[None, :] / 1e-3) ** slopes[:, None]
    integral = trapezoid((shapes / lisa_noise_h2(f)) ** 2, f, axis=1)
    amplitudes = threshold / np.sqrt(years * YEAR_SECONDS * integral)
    return slopes, amplitudes


def lisa_plis_h2(frequency_Hz, *, years=4.0, threshold=10.0):
    """Curva visual para espectros; a seleção quantitativa continua usando PIS acústico."""
    f = np.asarray(frequency_Hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Frequências devem ser positivas e finitas.")
    slopes, amplitudes = _plis_coefficients(years, threshold)
    return np.max(
        amplitudes[:, None] * (f.ravel()[None, :] / 1e-3) ** slopes[:, None], axis=0
    ).reshape(f.shape)
