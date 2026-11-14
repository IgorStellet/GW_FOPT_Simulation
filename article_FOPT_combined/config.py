"""Uma configuração imutável por campanha; recursos do job ficam fora dela."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal


def axis(start, stop, step):
    """Grade decimal inclusiva, com extremos separados por um número inteiro de passos."""
    if not all(math.isfinite(float(v)) for v in (start, stop, step)):
        raise ValueError("Extremos e passo devem ser finitos.")
    a, b, h = map(lambda v: Decimal(str(v)), (start, stop, step))
    if h <= 0 or b < a or (b - a) % h:
        raise ValueError("Grade exige passo>0, máximo>=mínimo e intervalo múltiplo do passo.")
    return tuple(float(a + i * h) for i in range(int((b - a) / h) + 1))


@dataclass(frozen=True)
class Numerics:
    """Precisão física/numerical; os valores efetivos acompanham cada ponto salvo."""

    T_min: float = 1.0
    T_max: float = 200.0
    phi_max: float = 1000.0
    n_phi: int = 1200
    n_T_seeds: int = 5
    x_eps: float = 0.01
    T_eps: float = 0.02
    phase_delta: float = 0.1
    phase_step: float = 0.001
    minima_tolerance: float = 1e-5
    nucleation_target: float = 140.0
    action_tolerance: float = 0.5
    root_T_tolerance: float = 1e-5
    beta_step_GeV: float = 0.5
    beta_min_step_GeV: float = 0.01
    beta_relative_tolerance: float = 0.2
    adaptive_beta: bool = True
    strong_threshold: float = 1.0
    g_star: float = 106.75
    wall_velocity: float = 1.0
    retry_resolution: bool = True

    def validate(self):
        for name, value in asdict(self).items():
            if name in ("retry_resolution", "adaptive_beta"):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} deve ser booleano.")
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser positivo e finito.")
        if (
            self.T_max <= self.T_min
            or self.n_phi < 100
            or self.n_T_seeds < 2
            or not isinstance(self.n_phi, int)
            or not isinstance(self.n_T_seeds, int)
        ):
            raise ValueError("T_max>T_min, n_phi inteiro>=100 e n_T_seeds inteiro>=2.")
        if self.wall_velocity > 1 or self.action_tolerance >= self.nucleation_target:
            raise ValueError("v_w<=1 e tolerância de ação menor que o alvo.")
        if self.beta_min_step_GeV > self.beta_step_GeV or self.beta_relative_tolerance >= 1:
            raise ValueError("Passo mínimo de beta<=passo pedido e tolerância relativa<1.")
        if self.adaptive_beta and self.beta_step_GeV < 2 * self.beta_min_step_GeV:
            raise ValueError("A validação adaptativa exige beta_step>=2*beta_min_step.")


@dataclass(frozen=True)
class Campaign:
    """Grade de fronteiras e amostragem entre bordas para as seis figuras.

    interior_points conta apenas os pontos internos: N pontos j/(N+1),
    j=1..N, mais as duas testemunhas internas das bordas. interior_stride=5
    amostra observáveis a cada quinta coluna de fronteira (25 GeV no padrão).
    """

    m6_range: tuple = (500.0, 2000.0, 5.0)
    m8_values: tuple = (math.inf, 840.8964152537145, 668.740304976422, 400.0)
    C_range: tuple = (0.0, 4.2)
    C_tolerance: float = 0.02
    seed_step: float = 0.2
    seed_resolution: float = 0.02
    walk_step: float = 0.15
    interior_points: int = 9
    interior_stride: int = 5
    numerics: Numerics = field(default_factory=Numerics)

    def validate(self):
        masses = axis(*self.m6_range)
        if masses[0] <= 0 or not self.m8_values or len(set(self.m8_values)) != len(self.m8_values):
            raise ValueError("m6 positivo e cenários m8 distintos.")
        if any(math.isnan(x) or x <= 0 for x in self.m8_values):
            raise ValueError("m8 deve ser positivo ou inf.")
        lo, hi = self.C_range
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi < (1000 / 246) ** 2):
            raise ValueError("Intervalo de C crescente, não negativo e dentro do domínio em v.")
        for value in (self.C_tolerance, self.seed_step, self.seed_resolution, self.walk_step):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Passos e tolerância de fronteira devem ser positivos.")
        if self.seed_resolution > self.seed_step or self.C_tolerance > hi - lo:
            raise ValueError("seed_resolution<=seed_step e C_tolerance<=amplitude de C.")
        if any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            for n in (self.interior_points, self.interior_stride)
        ):
            raise ValueError("Número de pontos e stride devem ser inteiros positivos.")
        self.numerics.validate()

    def masses(self):
        """Inclui o controle m6=inf; percorre massas finitas de 2000 para 500 GeV."""
        return (math.inf, *reversed(axis(*self.m6_range)))

    def observable_masses(self):
        """Subgrade determinística, incluindo ambos os extremos e o controle puro."""
        masses = self.masses()
        finite = masses[1:]
        return (math.inf, *dict.fromkeys((*finite[:: self.interior_stride], finite[-1])))


def campaign_from_dict(data):
    """Reconstrói tipos a partir do manifesto JSON estrito, onde inf é uma string."""
    values = dict(data)
    values["m6_range"] = tuple(values["m6_range"])
    values["C_range"] = tuple(values["C_range"])
    values["m8_values"] = tuple(math.inf if v == "inf" else float(v) for v in values["m8_values"])
    values["numerics"] = Numerics(**values["numerics"])
    campaign = Campaign(**values)
    campaign.validate()
    return campaign
