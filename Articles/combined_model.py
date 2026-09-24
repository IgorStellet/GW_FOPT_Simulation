"""Potencial combinado do artigo, com entradas em GeV.

A parte polinomial é

    V_tree = -mu² phi²/2 + lambda phi⁴/4 + phi⁶/(8 m6²) + phi⁸/(16 m8⁴).

Uma massa ``math.inf`` desliga o respectivo operador. Essas escalas de
supressão não determinam, por si sós, um cutoff de validade da EFT.

Preservamos as prescrições de ``thesis_gw_results.py``: termo de medida com
subtração local em v, Coleman–Weinberg com escala Q=v e contratermos finitos,
e funções térmicas spline de ``src/CosmoTransitions``. A ressoma é fixada na
aproximação de gauge da tese: um termo do tipo Arnold–Espinosa com três
modos longitudinais efetivos. Não é uma ressoma completa do setor escalar
ou da mistura longitudinal Z/fóton. As massas escalares usadas em CW e nas
funções térmicas vêm exclusivamente do potencial polinomial; a medida é
uma contribuição aditiva. Essas escolhas definem o modelo deste scan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from CosmoTransitions.finiteT import Jb, Jf
from CosmoTransitions.generic_potential import scalar_to_vector_potential_1d
from CosmoTransitions.helper_functions import gradientFunction, hessianFunction

RESUMMATION_SCHEME = "fixed_thesis_gauge_Arnold_Espinosa"
GOLDSTONE_SCHEME = "real_log_with_finite_step_zeroT_counterterms"


@dataclass(frozen=True)
class ModelParameters:
    """Entradas físicas; m6 e m8 positivos, ou +infinito para desligá-los.

    ``C`` é adimensional. ``Lambda_GeV`` pertence ao termo de medida.
    O domínio real exige C v²/Lambda² < 1. Todos os demais nomes terminados
    em ``_GeV`` representam massas ou campos medidos em GeV.
    """

    m6_GeV: float = 1000.0
    m8_GeV: float = math.inf
    C: float = 0.0
    Lambda_GeV: float = 1000.0
    v_GeV: float = 246.0
    mh_GeV: float = 125.0
    mw_GeV: float = 80.36
    mz_GeV: float = 91.19
    mt_GeV: float = 173.1

    def __post_init__(self) -> None:
        for name in ("m6_GeV", "m8_GeV"):
            value = float(getattr(self, name))
            if math.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} deve ser positivo ou math.inf.")
        for name in ("Lambda_GeV", "v_GeV", "mh_GeV", "mw_GeV", "mz_GeV", "mt_GeV"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} deve ser positivo e finito.")
        if not math.isfinite(self.C) or self.C < 0.0:
            raise ValueError("C deve ser não negativo e finito.")
        if self.mz_GeV < self.mw_GeV:
            raise ValueError("mz_GeV deve ser maior ou igual a mw_GeV.")
        if self.measure_t0 >= 1.0:
            raise ValueError(
                "O vácuo eletrofraco está fora do domínio real do logaritmo."
            )

    @property
    def inverse_m6_squared(self) -> float:
        """Coeficiente do operador de dimensão seis, em GeV⁻²."""
        return (1.0 / self.m6_GeV) ** 2

    @property
    def inverse_m8_fourth(self) -> float:
        """Coeficiente do operador de dimensão oito, em GeV⁻⁴."""
        return (1.0 / self.m8_GeV) ** 4

    @property
    def g(self) -> float:
        return 2.0 * self.mw_GeV / self.v_GeV

    @property
    def gp(self) -> float:
        return math.sqrt(4.0 * self.mz_GeV**2 / self.v_GeV**2 - self.g**2)

    @property
    def yt(self) -> float:
        return math.sqrt(2.0) * self.mt_GeV / self.v_GeV

    @property
    def measure_t0(self) -> float:
        return self.C * (self.v_GeV / self.Lambda_GeV) ** 2

    @property
    def lambda_tree(self) -> float:
        """Escolha que impõe V_tree''(v)=mh² e V_tree'(v)=0."""
        v = self.v_GeV
        return (
            self.mh_GeV**2 / (2.0 * v**2)
            - 1.5 * self.inverse_m6_squared * v**2
            - 1.5 * self.inverse_m8_fourth * v**4
        )

    @property
    def mu2_tree_GeV2(self) -> float:
        v = self.v_GeV
        return (
            self.lambda_tree * v**2
            + 0.75 * self.inverse_m6_squared * v**4
            + 0.5 * self.inverse_m8_fourth * v**6
        )


class CombinedPotential:
    """Potencial V(phi,T), em GeV⁴, e adaptador Vtot(X,T) para o núcleo.

    O objeto aceita escalares ou arrays compatíveis por broadcasting.
    ``Vtot`` usa a convenção X[..., 0] de um único campo. Fora do domínio
    real do logaritmo devolvemos NaN, nunca a continuação log|1-t|.

    Os contratermos são calculados uma vez por ponto físico. Não há aqui
    implementação paralela de derivadas ou funções térmicas do ``src``.
    """

    def __init__(self, parameters: ModelParameters):
        self.parameters = parameters
        self.Vtot = scalar_to_vector_potential_1d(self)
        self.counterterm_step_GeV = max(1.0e-3 * parameters.v_GeV, 1.0e-2)
        # Mesma regularização operacional do Goldstone usada na tese.
        # Sua curvatura CW em v é singular no limite h -> 0; h é parte da
        # prescrição, não uma tolerância que deva ser reduzida sem estudo.
        raw_vector = scalar_to_vector_potential_1d(lambda phi, T: self.cw_raw(phi))
        gradient = gradientFunction(raw_vector, self.counterterm_step_GeV, 1, order=2)
        hessian = hessianFunction(raw_vector, self.counterterm_step_GeV, 1, order=2)
        v = parameters.v_GeV
        first = float(np.asarray(gradient([v], 0.0)).ravel()[0])
        second = float(np.asarray(hessian([v], 0.0)).ravel()[0])
        self.delta_lambda = (first / v - second) / (2.0 * v**2)
        self.delta_mu2_GeV2 = first / v + self.delta_lambda * v**2

    @property
    def domain_limit(self) -> float:
        """Limite |phi| < Lambda/sqrt(C); infinito quando C=0."""
        p = self.parameters
        return p.Lambda_GeV / math.sqrt(p.C) if p.C > 0.0 else math.inf

    def tree(self, phi: ArrayLike) -> np.ndarray:
        p = self.parameters
        x2 = np.asarray(phi, dtype=float) ** 2
        return (
            -0.5 * p.mu2_tree_GeV2 * x2
            + 0.25 * p.lambda_tree * x2**2
            + 0.125 * p.inverse_m6_squared * x2**3
            + 0.0625 * p.inverse_m8_fourth * x2**4
        )

    def field_masses_squared(self, phi: ArrayLike) -> dict[str, np.ndarray]:
        """Massas de h, três Goldstones, W, Z e top, em GeV².

        A medida não altera estas massas na prescrição herdada da tese.
        Massas escalares negativas são mantidas: as funções spline térmicas
        implementam a parte real da continuação correspondente.
        """
        p = self.parameters
        x2 = np.asarray(phi, dtype=float) ** 2
        a, b = p.inverse_m6_squared, p.inverse_m8_fourth
        return {
            "h": -p.mu2_tree_GeV2
            + 3.0 * p.lambda_tree * x2
            + 3.75 * a * x2**2
            + 3.5 * b * x2**3,
            "chi": -p.mu2_tree_GeV2
            + p.lambda_tree * x2
            + 0.75 * a * x2**2
            + 0.5 * b * x2**3,
            "W": 0.25 * p.g**2 * x2,
            "Z": 0.25 * (p.g**2 + p.gp**2) * x2,
            "top": 0.5 * p.yt**2 * x2,
        }

    def cw_raw(self, phi: ArrayLike) -> np.ndarray:
        """Coleman–Weinberg real em MS-bar, com Q=v e todos os loops da tese."""
        result = np.zeros_like(np.asarray(phi, dtype=float))
        q2 = self.parameters.v_GeV**2
        masses = self.field_masses_squared(phi)
        for name, dof, constant in (
            ("h", 1.0, 1.5),
            ("chi", 3.0, 1.5),
            ("W", 6.0, 5.0 / 6.0),
            ("Z", 3.0, 5.0 / 6.0),
            ("top", -12.0, 1.5),
        ):
            m2 = masses[name]
            # A multiplicação por m2² dá o limite zero para m2=0.
            real_log = np.log(np.maximum(np.abs(m2), 1.0e-300) / q2)
            result += dof * m2**2 * (real_log - constant) / (64.0 * np.pi**2)
        return result

    def counterterms(self, phi: ArrayLike) -> np.ndarray:
        """Contratermos locais; a constante aditiva é fixada em zero."""
        x2 = np.asarray(phi, dtype=float) ** 2
        return -0.5 * self.delta_mu2_GeV2 * x2 + 0.25 * self.delta_lambda * x2**2

    def measure(self, phi: ArrayLike) -> np.ndarray:
        """Medida logarítmica subtraída: derivadas primeira e segunda nulas em v."""
        p = self.parameters
        x = np.asarray(phi, dtype=float)
        if p.C == 0.0:
            return np.zeros_like(x)
        t = p.C * (x / p.Lambda_GeV) ** 2
        t0 = p.measure_t0
        polynomial = ((1.0 - 2.0 * t0) * t + 0.5 * t**2) / (1.0 - t0) ** 2
        # log1p conserva precisão para C pequeno; nenhum clipping estende
        # artificialmente o domínio físico através da singularidade.
        with np.errstate(invalid="ignore", divide="ignore"):
            value = -(p.Lambda_GeV**4) * (np.log1p(-t) + polynomial) / (8.0 * np.pi**2)
        return np.where((np.abs(x) < self.domain_limit) & (t < 1.0), value, np.nan)

    def thermal(self, phi: ArrayLike, T: ArrayLike) -> np.ndarray:
        """Correção térmica: spline recebe m²/T², e não m/T."""
        x, temperature = np.broadcast_arrays(
            np.asarray(phi, dtype=float), np.asarray(T, dtype=float)
        )
        if np.any(temperature < 0.0):
            raise ValueError("T deve ser não negativa.")
        result = np.zeros_like(x)
        positive = temperature > 0.0
        if not np.any(positive):
            return result
        temp = temperature[positive]
        masses = self.field_masses_squared(x[positive])
        thermal_sum = sum(
            dof * Jb(masses[name] / temp**2, approx="spline")
            for name, dof in (("h", 1.0), ("chi", 3.0), ("W", 6.0), ("Z", 3.0))
        )
        thermal_sum += 12.0 * Jf(masses["top"] / temp**2, approx="spline")
        result[positive] = temp**4 * thermal_sum / (2.0 * np.pi**2)
        return result

    def daisy(self, phi: ArrayLike, T: ArrayLike) -> np.ndarray:
        """Ressoma fixa de gauge da tese (três modos efetivos).

        g_eff² = 4(mW²+mZ²)/(3v²), mL²=g_eff² phi²/4,
        Pi_L=11 g_eff² T²/6 e V_ring=-3T[(mL²+Pi_L)^(3/2)
        -(mL²)^(3/2)]/(12 pi). Não acrescentamos massas de Debye escalares.
        """
        p = self.parameters
        x, temperature = np.broadcast_arrays(
            np.asarray(phi, dtype=float), np.asarray(T, dtype=float)
        )
        if np.any(temperature < 0.0):
            raise ValueError("T deve ser não negativa.")
        g_eff2 = 4.0 * (p.mw_GeV**2 + p.mz_GeV**2) / (3.0 * p.v_GeV**2)
        longitudinal_m2 = 0.25 * g_eff2 * x**2
        debye_m2 = (11.0 / 6.0) * g_eff2 * temperature**2
        return (
            -temperature
            * 3.0
            * ((longitudinal_m2 + debye_m2) ** 1.5 - longitudinal_m2**1.5)
            / (12.0 * np.pi)
        )

    def __call__(self, phi: ArrayLike, T: ArrayLike) -> np.ndarray:
        """Soma das contribuições, sem mudar a normalização em função de T."""
        return (
            self.tree(phi)
            + self.cw_raw(phi)
            + self.counterterms(phi)
            + self.measure(phi)
            + self.thermal(phi, T)
            + self.daisy(phi, T)
        )

    def zero_temperature_diagnostics(
        self, *, phi_max: float = 1000.0, n_grid: int = 1201
    ) -> dict[str, Any]:
        """Verificações locais em v e busca de mínimos numa janela declarada.

        ``phi_max`` é um limite numérico, não um cutoff EFT inferido de m6
        ou m8. A janela é encurtada a 0.99 do ramo logarítmico se necessário.
        Mínimos interiores vistos na malha são refinados; a classificação
        resultante refere-se somente a essa janela e resolução.
        """
        p = self.parameters
        upper = min(float(phi_max), 0.99 * self.domain_limit)
        if not math.isfinite(upper) or upper <= p.v_GeV:
            raise ValueError("A janela de diagnóstico deve ser finita e conter v.")
        if n_grid < 5:
            raise ValueError("n_grid deve ser pelo menos 5.")
        phi = np.linspace(0.0, upper, int(n_grid))
        values = self(phi, 0.0)
        if not np.all(np.isfinite(values)):
            raise ValueError("Potencial não finito na janela de diagnóstico T=0.")
        candidates = [
            (0.0, float(values[0])),
            (upper, float(values[-1])),
            (p.v_GeV, float(self(p.v_GeV, 0.0))),
        ]
        interior = (
            np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))
            + 1
        )
        for index in interior:
            minimum = minimize_scalar(
                lambda x: float(self(x, 0.0)),
                bounds=(phi[index - 1], phi[index + 1]),
                method="bounded",
            )
            if minimum.success:
                candidates.append((float(minimum.x), float(minimum.fun)))
        minimum_phi, minimum_value = min(candidates, key=lambda pair: pair[1])
        h = self.counterterm_step_GeV
        gradient = gradientFunction(self.Vtot, h, 1, order=2)
        hessian = hessianFunction(self.Vtot, h, 1, order=2)
        Vv, V0 = float(self(p.v_GeV, 0.0)), float(values[0])
        # Tolerância energética explícita evita classificar ruído de
        # arredondamento do mínimo refinado como outro vácuo mais profundo.
        energy_tolerance = 1.0e-8 * max(p.v_GeV**4, abs(Vv), abs(minimum_value), 1.0)
        return {
            "resummation_scheme": RESUMMATION_SCHEME,
            "goldstone_scheme": GOLDSTONE_SCHEME,
            "lambda_tree": p.lambda_tree,
            "mu2_tree_GeV2": p.mu2_tree_GeV2,
            "delta_lambda": self.delta_lambda,
            "delta_mu2_GeV2": self.delta_mu2_GeV2,
            "counterterm_step_GeV": h,
            "domain_limit_GeV": self.domain_limit,
            "scan_phi_max_GeV": upper,
            "scan_grid_points": int(n_grid),
            "dV_at_v_GeV3": float(np.asarray(gradient([p.v_GeV], 0.0)).ravel()[0]),
            "d2V_at_v_GeV2": float(np.asarray(hessian([p.v_GeV], 0.0)).ravel()[0]),
            "mh2_target_GeV2": p.mh_GeV**2,
            "V_at_v_GeV4": Vv,
            "V_at_origin_GeV4": V0,
            "ew_below_origin": Vv < V0,
            "ew_is_lowest_sampled": Vv <= minimum_value + energy_tolerance,
            "minimum_phi_GeV": minimum_phi,
            "minimum_V_GeV4": minimum_value,
            "energy_tolerance_GeV4": energy_tolerance,
        }
