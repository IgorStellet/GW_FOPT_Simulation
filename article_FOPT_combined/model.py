"""Potencial do artigo em GeV e GeV⁴.

O potencial polinomial do Higgs e a medida funcional definem o setor escalar.
As correções diretas de um loop usam W, Z e top. A ressoma Arnold–Espinosa
usa os modos longitudinais de gauge com a blindagem do plasma eletrofraco.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy

from .core import finiteT, generic

MODEL_VERSION = "combined_WZt_AE_gauge_v1"


@dataclass(frozen=True)
class Parameters:
    """Três coordenadas físicas; inf desliga o respectivo operador polinomial."""

    m6_GeV: float = math.inf
    m8_GeV: float = math.inf
    C: float = 0.0

    def __post_init__(self):
        # As mesmas coordenadas têm o mesmo identificador, mesmo se a API
        # receber 1000 em uma chamada e 1000.0 na próxima.
        for name in ("m6_GeV", "m8_GeV", "C"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for mass in (self.m6_GeV, self.m8_GeV):
            if math.isnan(mass) or mass <= 0:
                raise ValueError("m6 e m8 devem ser positivos ou inf.")
        if not math.isfinite(self.C) or not 0 <= self.C < 1000.0**2 / 246.0**2:
            raise ValueError("C deve ser não negativo e manter v dentro do domínio logarítmico.")


class Potential:
    """V(phi,T), vetorizado, com todos os termos acessíveis separadamente.

    Lambda e as entradas eletrofracas são fixas. O Coleman–Weinberg inclui
    os contratermos finitos que preservam o mínimo e a curvatura em v.
    """

    v = 246.0
    mh, mw, mz, mt = 125.0, 80.36, 91.19, 173.1
    Lambda = 1000.0

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.a6, self.a8 = (1 / parameters.m6_GeV) ** 2, (1 / parameters.m8_GeV) ** 4
        self.g = 2 * self.mw / self.v
        self.gp = math.sqrt(4 * self.mz**2 / self.v**2 - self.g**2)
        self.yt = math.sqrt(2) * self.mt / self.v
        self.lam = (
            self.mh**2 / (2 * self.v**2) - 1.5 * self.a6 * self.v**2 - 1.5 * self.a8 * self.v**4
        )
        self.mu2 = self.lam * self.v**2 + 0.75 * self.a6 * self.v**4 + 0.5 * self.a8 * self.v**6
        self._cw_amplitude = (6 * self.mw**4 + 3 * self.mz**4 - 12 * self.mt**4) / (64 * np.pi**2)
        self.Vtot = generic.scalar_to_vector_potential_1d(self)

    @property
    def domain_limit(self):
        """Limite estrito |phi|<Lambda/sqrt(C); infinito em C=0."""
        return self.Lambda / math.sqrt(self.parameters.C) if self.parameters.C else math.inf

    def tree(self, phi):
        """V0=-mu² phi²/2+lambda phi⁴/4+phi⁶/(8m6²)+phi⁸/(16m8⁴)."""
        s = np.asarray(phi, dtype=float) ** 2
        return -self.mu2 * s / 2 + self.lam * s**2 / 4 + self.a6 * s**3 / 8 + self.a8 * s**4 / 16

    def masses_squared(self, phi):
        """Massas de W, Z e top em GeV², para o campo de fundo phi."""
        s = np.asarray(phi, dtype=float) ** 2
        return {
            "W": self.g**2 * s / 4,
            "Z": (self.g**2 + self.gp**2) * s / 4,
            "top": self.yt**2 * s / 2,
        }

    def coleman_weinberg(self, phi):
        """CW renormalizado de W(6), Z(3), top(-12), com V'(v)=V''(v)=0.

        Como as massas ao quadrado são proporcionais a phi², a subtração é
        analítica. xlogy implementa o limite s² log(s) = 0 em s = 0.
        """
        s = (np.asarray(phi, dtype=float) / self.v) ** 2
        return self._cw_amplitude * (xlogy(s**2, s) - 1.5 * s**2 + 2 * s)

    def measure(self, phi):
        """Medida logarítmica com primeira/segunda derivadas subtraídas em v."""
        x = np.asarray(phi, dtype=float)
        if self.parameters.C == 0:
            return np.zeros_like(x)
        t = self.parameters.C * x**2 / self.Lambda**2
        t0 = self.parameters.C * self.v**2 / self.Lambda**2
        polynomial = ((1 - 2 * t0) * t + t**2 / 2) / (1 - t0) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            value = -(self.Lambda**4) * (np.log1p(-t) + polynomial) / (8 * np.pi**2)
        return np.where(t < 1, value, np.nan)

    def thermal(self, phi, temperature):
        """Jb/Jf do src por spline em m²/T²; conserva as massas não ressomadas."""
        x, T = np.broadcast_arrays(
            np.asarray(phi, dtype=float), np.asarray(temperature, dtype=float)
        )
        if np.any(~np.isfinite(T)) or np.any(T < 0):
            raise ValueError("Temperatura deve ser finita e não negativa.")
        result = np.zeros_like(x)
        positive = T > 0
        if np.any(positive):
            tp = T[positive]
            masses = self.masses_squared(x[positive])
            total = sum(
                n * finiteT.Jb(masses[k] / tp**2, approx="spline") for k, n in (("W", 6), ("Z", 3))
            )
            # O Jf do núcleo já inclui o sinal fermiônico.
            total += 12 * finiteT.Jf(masses["top"] / tp**2, approx="spline")
            result[positive] = tp**4 * total / (2 * np.pi**2)
        return result

    def longitudinal_masses_squared(self, phi, temperature):
        """W_L e autovalores neutros Z_L/gamma_L da matriz térmica eletrofraca."""
        s, T2 = np.asarray(phi, dtype=float) ** 2, np.asarray(temperature, dtype=float) ** 2
        a = self.g**2 * s / 4 + 11 * self.g**2 * T2 / 6
        d = self.gp**2 * s / 4 + 11 * self.gp**2 * T2 / 6
        off = -self.g * self.gp * s / 4
        gap = np.sqrt((a - d) ** 2 + 4 * off**2)
        # O menor autovalor pode ficar ligeiramente negativo por arredondamento.
        return a, (a + d + gap) / 2, np.maximum((a + d - gap) / 2, 0)

    def gauge_rings(self, phi, temperature):
        """Dois modos W_L, um Z_L e um gamma_L; modos transversais permanecem nus."""
        masses = self.masses_squared(phi)
        w, z, photon = self.longitudinal_masses_squared(phi, temperature)
        return (
            -np.asarray(temperature)
            / (12 * np.pi)
            * (2 * (w**1.5 - masses["W"] ** 1.5) + z**1.5 - masses["Z"] ** 1.5 + photon**1.5)
        )

    def __call__(self, phi, temperature):
        """Soma física única; todas as derivadas do núcleo recebem esta mesma soma."""
        return (
            self.tree(phi)
            + self.coleman_weinberg(phi)
            + self.measure(phi)
            + self.thermal(phi, temperature)
            + self.gauge_rings(phi, temperature)
        )

    def diagnostics(self):
        """Registra normalização e condição necessária de vácuo eletrofraco a T=0.

        V(v)<V(0) é necessário para a seleção cosmológica adotada. Esta consulta
        não certifica ausência de outros mínimos fora da janela examinada.
        """
        delta = float(self(self.v, 0) - self(0, 0))
        return {
            "model_version": MODEL_VERSION,
            "lambda_tree": self.lam,
            "mu2_tree_GeV2": self.mu2,
            "Vv_minus_V0_GeV4": delta,
            "ew_below_origin": delta < 0,
            "log_domain_limit_GeV": self.domain_limit,
        }
