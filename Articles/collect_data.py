"""Coleta reprodutível do artigo sobre mecanismos combinados de FOPT.

Execute na raiz: ``python -m Articles.collect_data --dry-run``.
O módulo só coordena os cálculos: integrais térmicas, fases, bounce,
derivadas e ondas gravitacionais continuam pertencendo a CosmoTransitions.
Nenhuma figura é produzida. Consulte README.md para convenções e exemplos.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import multiprocessing
import os
import platform
import sqlite3
import subprocess
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Um processo por ponto já fornece paralelismo. Evita que cada processo abra
# também um conjunto grande de threads BLAS e concorra pelos mesmos recursos.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

from Articles.combined_model import CombinedPotential, ModelParameters
from CosmoTransitions.generic_potential import (
    _build_phases_and_transitions,
    build_finite_T_derivatives,
)
from CosmoTransitions.gravitational_Waves import (
    GravitationalWaveCalculator,
    bbo_sensitivity_s_pis,
    decigo_sensitivity_s_pis,
    gw_f_sw_peak,
    gw_omega_sw_h2,
    lisa_sensitivity_s_pis,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
M8_SCENARIOS_GEV = (math.inf, 840.8964152537145, 668.740304976422)


@dataclass(frozen=True)
class Settings:
    """Escolhas numéricas e fenomenológicas comuns a TODOS os pontos."""

    T_min: float = 1.0
    T_max: float = 250.0
    phi_max: float = 1000.0  # Limite de busca, não um cutoff UV inferido.
    n_phi: int = 1200
    n_T_seeds: int = 5
    deltaX_target: float = 0.1
    x_eps: float = 1e-3
    T_eps: float = 1e-2
    potential_derivative_order: int = 4
    minima_phitol: float = 1e-5
    nucleation_target: float = 140.0
    action_tolerance: float = 0.5  # Adimensional: |S3/T - 140|.
    root_T_tolerance: float = 1e-5  # GeV; salvaguarda distinta do residual.
    root_maxiter: int = 100
    beta_step: float = 0.5  # GeV; não encolher silenciosamente junto a Tc.
    beta_check: bool = False
    beta_relative_tolerance: float = 0.2
    g_star: float = 106.75
    wall_velocity: float = 1.0
    strong_threshold: float = 1.0  # Convenção operacional |delta phi|/Tn.

    def validate(self) -> None:
        positive = (
            "T_min",
            "T_max",
            "phi_max",
            "deltaX_target",
            "x_eps",
            "T_eps",
            "minima_phitol",
            "nucleation_target",
            "action_tolerance",
            "root_T_tolerance",
            "beta_step",
            "beta_relative_tolerance",
            "g_star",
            "wall_velocity",
            "strong_threshold",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser finito e positivo.")
        if self.T_max <= self.T_min or self.n_phi < 20 or self.n_T_seeds < 2:
            raise ValueError("Exige T_max>T_min, n_phi>=20 e n_T_seeds>=2.")
        if self.wall_velocity > 1 or self.action_tolerance >= self.nucleation_target:
            raise ValueError(
                "Velocidade <=1 e tolerância menor que o alvo de nucleação."
            )


def inclusive_axis(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Grade decimal inclusiva, sem deriva de arredondamento de np.arange."""
    if not all(math.isfinite(v) for v in (start, stop, step)):
        raise ValueError("Extremos e passo da grade devem ser finitos.")
    a, b, h = (Decimal(str(v)) for v in (start, stop, step))
    if h <= 0 or b < a:
        raise ValueError("Grade exige passo>0 e máximo>=mínimo.")
    count = (b - a) / h
    if count != count.to_integral_value():
        raise ValueError("O intervalo deve ser múltiplo inteiro do passo.")
    return tuple(float(a + i * h) for i in range(int(count) + 1))


def scan_points(masses, couplings, m8_values, baselines=True):
    """Inclui C=0 e m6=inf para comparações puras, sem repetir coordenadas."""
    cs = tuple(sorted(set(couplings) | ({0.0} if baselines else set())))
    ms = tuple(sorted(set(masses) | ({math.inf} if baselines else set())))
    m8s = tuple(dict.fromkeys(m8_values))
    if baselines and math.inf not in m8s:
        m8s += (math.inf,)  # Garante a referência Glauber puro.
    for m8, m6, C in itertools.product(m8s, ms, cs):
        yield ModelParameters(m6_GeV=m6, m8_GeV=m8, C=C, Lambda_GeV=1000.0)


def json_safe(value):
    """JSON estrito: massas desligadas são 'inf'; valores ausentes são null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value == math.inf else None
    return value


def canonical_json(value) -> str:
    return json.dumps(
        json_safe(value), sort_keys=True, allow_nan=False, ensure_ascii=False
    )


def point_id(params: ModelParameters) -> str:
    return hashlib.sha256(canonical_json(asdict(params)).encode()).hexdigest()[:24]


def mechanism(params: ModelParameters) -> str:
    polynomial = math.isfinite(params.m6_GeV) or math.isfinite(params.m8_GeV)
    return (
        ("combined" if polynomial else "measure_only")
        if params.C
        else ("polynomial_only" if polynomial else "SM_reference")
    )


class NucleationCriterion:
    """Encerra a busca quando o residual FÍSICO está dentro da tolerância.

    A faixa nula é deliberada: não se exige uma raiz artificialmente precisa
    de ações calculadas numericamente. As amostras cruas permanecem salvas.
    """

    def __init__(self, target=140.0, tolerance=0.5):
        self.target, self.tolerance = target, tolerance
        self.samples = {}

    def __call__(self, action, temperature):
        residual = action / temperature - self.target
        self.samples[float(temperature)] = {
            "T_GeV": float(temperature),
            "S3_GeV": float(action),
            "S3_over_T": float(action / temperature),
            "residual": float(residual),
        }
        return 0.0 if abs(residual) <= self.tolerance else residual


def transition_record(transition, phases, derivatives, settings):
    """Transforma UMA transição do histórico em observáveis e dados de auditoria."""
    s = settings
    Tn = float(transition["Tnuc"])
    row = {
        "status": "not_first_order",
        "trantype": int(transition["trantype"]),
        "Tn_GeV": Tn,
        "high_phase": str(transition["high_phase"]),
        "low_phase": str(transition["low_phase"]),
        "beta_order": 2,
        "beta_step_GeV": s.beta_step,
        "g_star": s.g_star,
        "v_w": s.wall_velocity,
    }
    critical = transition.get("crit_trans") or {}
    row["Tc_GeV"] = critical.get("Tcrit")
    if row["Tc_GeV"] is not None:
        for side in ("high", "low"):
            if f"{side}_vev" in critical:
                row[f"phi_{side}_Tc_GeV"] = float(
                    np.asarray(critical[f"{side}_vev"]).ravel()[0]
                )
    details = {"quality_flags": [], "action_samples": []}
    if row["trantype"] != 1:
        return row, details
    row["status"] = "nucleated"
    row["S3_GeV"] = float(transition["action"])
    row["S3_over_T"] = row["S3_GeV"] / Tn
    row["nucleation_residual"] = row["S3_over_T"] - s.nucleation_target
    row["nucleation_accepted"] = abs(row["nucleation_residual"]) <= s.action_tolerance
    if not row["nucleation_accepted"]:
        details["quality_flags"].append("nucleation_residual_outside_tolerance")

    calculator = None
    try:
        calculator = GravitationalWaveCalculator(
            derivatives.V,
            derivatives.gradV,
            derivatives.dV_dT,
            phases,
            transition["high_phase"],
            transition["low_phase"],
            minima_phitol=s.minima_phitol,
        )
        thermo = calculator.thermodynamics(Tn, s.g_star)
        row.update(
            {k: v for k, v in thermo.items() if k not in ("phi_high", "phi_low")}
        )
        row["phi_high_GeV"] = float(np.asarray(thermo["phi_high"]).ravel()[0])
        row["phi_low_GeV"] = float(np.asarray(thermo["phi_low"]).ravel()[0])
        row["delta_phi_over_Tn"] = abs(row["phi_low_GeV"] - row["phi_high_GeV"]) / Tn
        row["strong_by_field_ratio"] = row["delta_phi_over_Tn"] >= s.strong_threshold
        if row["Tc_GeV"]:
            row["supercooling_fraction"] = 1.0 - Tn / row["Tc_GeV"]
        # A central derivative requires metastability on BOTH sides. If the
        # requested stencil crosses Tc/a phase endpoint, leave beta unresolved.
        lower = max(
            float(p.T[0]) for p in (calculator.high_phase, calculator.low_phase)
        )
        upper = min(
            float(p.T[-1]) for p in (calculator.high_phase, calculator.low_phase)
        )
        if row["Tc_GeV"] is not None:
            upper = min(upper, float(row["Tc_GeV"]))
        row.update(phase_overlap_min_GeV=lower, phase_overlap_max_GeV=upper)
        if Tn - s.beta_step <= lower or Tn + s.beta_step >= upper:
            raise ValueError("beta_stencil_outside_metastable_interval")
        row["beta_over_H"] = calculator.beta_over_H(Tn, s.beta_step, order=2)
        row["beta_Tminus_GeV"] = Tn - s.beta_step
        row["beta_Tplus_GeV"] = Tn + s.beta_step
        row["beta_Fminus"] = calculator._S3_cache[Tn - s.beta_step] / (Tn - s.beta_step)
        row["beta_Fplus"] = calculator._S3_cache[Tn + s.beta_step] / (Tn + s.beta_step)
        if s.beta_check:
            check = calculator.beta_over_H(Tn, s.beta_step / 2, order=2)
            row["beta_over_H_half_step"] = check
            row["beta_relative_change"] = abs(check - row["beta_over_H"]) / max(
                abs(check), abs(row["beta_over_H"]), 1e-30
            )
            if row["beta_relative_change"] > s.beta_relative_tolerance:
                details["quality_flags"].append("beta_step_sensitivity")
        if not (math.isfinite(row["beta_over_H"]) and row["beta_over_H"] > 0):
            details["quality_flags"].append("nonpositive_or_nonfinite_beta")
        if not (math.isfinite(row["alpha_trace"]) and row["alpha_trace"] > 0):
            details["quality_flags"].append("nonpositive_or_nonfinite_alpha_trace")
        # Sem valor absoluto para esconder um sinal termodinâmico inesperado.
        # Os mesmos inputs permitem reconstruir todo espectro usando o src.
        if not details["quality_flags"]:
            inputs = {
                "alpha": row["alpha_trace"],
                "beta_over_H": row["beta_over_H"],
                "T_star": Tn,
                "g_star": s.g_star,
                "v_w": s.wall_velocity,
            }
            f_mHz = gw_f_sw_peak(row["beta_over_H"], Tn, s.g_star, s.wall_velocity)
            amplitude = float(gw_omega_sw_h2(np.array([f_mHz]), **inputs)[0])
            row.update(f_sw_peak_Hz=f_mHz * 1e-3, omega_sw_peak_h2=amplitude)
            for name, sensitivity in (
                ("LISA", lisa_sensitivity_s_pis),
                ("DECIGO", decigo_sensitivity_s_pis),
                ("BBO", bbo_sensitivity_s_pis),
            ):
                pis = float(sensitivity(np.array([f_mHz]))[0])
                row[f"{name}_sw_PIS_h2"] = pis
                row[f"{name}_sw_PIS_ratio"] = amplitude / pis
            details["spectrum_inputs"] = inputs
    except Exception as error:  # noqa: BLE001 - preserve point-level numerical failures
        details["quality_flags"].append(f"{type(error).__name__}: {error}")
        details["traceback"] = traceback.format_exc()
    finally:
        if calculator is not None:
            details["action_samples"] = [
                {"T_GeV": T, "S3_GeV": action, "S3_over_T": action / T}
                for T, action in sorted(calculator._S3_cache.items())
            ]
    if details["quality_flags"]:
        row["status"] = "observables_unresolved"
    row["quality_flags"] = "; ".join(details["quality_flags"])
    return row, details


def evaluate_point(params: ModelParameters, settings: Settings):
    """Executa um ponto isolado; exceções são dados, nunca ausência de FOPT."""
    started = time.perf_counter()
    row = {
        "point_id": point_id(params),
        **asdict(params),
        "mechanism": mechanism(params),
        "status": "numerical_failure",
        "n_phases": 0,
        "n_critical": 0,
        "n_first_order": 0,
        "n_transitions": 0,
    }
    details = {
        "parameters": asdict(params),
        "phases": {},
        "critical_transitions": [],
        "transitions": [],
        "warnings": [],
    }
    records = []
    criterion = NucleationCriterion(
        settings.nucleation_target, settings.action_tolerance
    )

    # Limita a captura de mensagens sem perder o início de diagnósticos úteis.
    class BoundedLog(io.StringIO):
        def write(self, text):
            remaining = max(0, 32000 - self.tell())
            super().write(text[:remaining])
            return len(text)

    log = BoundedLog()
    with (
        redirect_stdout(log),
        redirect_stderr(log),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("once")
        try:
            model = CombinedPotential(params)
            phi_max = min(settings.phi_max, 0.98 * model.domain_limit)
            if phi_max <= params.v_GeV:
                raise ValueError("Limite de busca deve incluir o vácuo eletrofraco.")
            row["phi_search_max_GeV"] = phi_max
            zero = model.zero_temperature_diagnostics(
                phi_max=phi_max, n_grid=settings.n_phi
            )
            details["zero_temperature"] = zero
            row.update(zero)
            derivatives = build_finite_T_derivatives(
                model.Vtot,
                Ndim=1,
                x_eps=settings.x_eps,
                T_eps=settings.T_eps,
                deriv_order=settings.potential_derivative_order,
                X_ref=np.array([0.0]),
            )

            def outside_domain(X):
                phi = float(np.asarray(X).ravel()[0])
                # Mantém a origem no interior da grade; descarta cópias negativas
                # do mesmo vácuo (o potencial radial é par).
                return phi < -settings.x_eps * 10 or phi >= phi_max

            summary = _build_phases_and_transitions(
                derivatives.V,
                derivatives.gradV,
                derivatives.hessV,
                derivatives.dV_dT,
                derivatives.dgradV_dT,
                T_min=settings.T_min,
                T_max=settings.T_max,
                phi_range=(-0.15 * phi_max, phi_max),
                n_phi_scan=settings.n_phi,
                n_T_seeds=settings.n_T_seeds,
                deltaX_target=settings.deltaX_target,
                forbidCrit=outside_domain,
                nuclCriterion=criterion,
                Tn_Ttol=settings.root_T_tolerance,
                Tn_maxiter=settings.root_maxiter,
                tunnelFromPhase_args={
                    "phitol": settings.minima_phitol,
                    "overlapAngle": 0.0,
                    "verbose": False,
                },
                verbose=False,
            )
            phases = summary["phases"]
            row["n_phases"] = len(phases)
            row["n_critical"] = len(summary["critical_transitions"])
            row["n_first_order_critical"] = sum(
                tr["trantype"] == 1 for tr in summary["critical_transitions"]
            )
            row["start_phase"] = str(summary["start_phase_key"])
            row["start_phi_GeV"] = float(
                np.asarray(summary["start_phase"].valAt(settings.T_max)).ravel()[0]
            )
            # Histórias completas permitem outros gráficos de fases sem tunneling.
            details["phases"] = {
                str(key): {"T_GeV": p.T, "phi_GeV": p.X, "dphi_dT": p.dXdT}
                for key, p in phases.items()
            }
            details["critical_transitions"] = summary["critical_transitions"]
            for index, transition in enumerate(summary["full_transitions"]):
                tr, extra = transition_record(transition, phases, derivatives, settings)
                tr.update(
                    point_id=row["point_id"],
                    transition_index=index,
                    m6_GeV=params.m6_GeV,
                    m8_GeV=params.m8_GeV,
                    C=params.C,
                    Lambda_GeV=params.Lambda_GeV,
                )
                if tr.get("phi_low_GeV") is not None:
                    phi = max(abs(tr["phi_low_GeV"]), abs(tr["phi_high_GeV"]))
                    tr["log_domain_margin"] = (
                        1 - params.C * phi**2 / params.Lambda_GeV**2
                    )
                records.append(tr)
                details["transitions"].append({"observables": tr, **extra})
            row["n_transitions"] = len(records)
            row["n_first_order"] = sum(r["trantype"] == 1 for r in records)
            row["status"] = (
                "nucleated" if row["n_first_order"] else "no_nucleation_found"
            )
            if not row["n_first_order"] and not row["n_first_order_critical"]:
                row["status"] = "no_first_order_found"
            if any(r["status"] == "observables_unresolved" for r in records):
                row["status"] = "observables_unresolved"
        except Exception as error:  # noqa: BLE001 - failures must become explicit records
            row["error_type"], row["error_message"] = type(error).__name__, str(error)
            details["traceback"] = traceback.format_exc()
        details["warnings"] = sorted({str(w.message) for w in caught})
    details["nucleation_samples"] = list(criterion.samples.values())
    details["log"] = log.getvalue()
    row["runtime_seconds"] = time.perf_counter() - started
    row["warning_count"] = len(details["warnings"])
    row["completed_utc"] = datetime.now(UTC).isoformat()
    return row, records, details


def provenance(settings, grid):
    """Hash do código efetivo, e não apenas do último commit."""
    paths = sorted((ROOT / "src/CosmoTransitions").glob("*.py")) + sorted(
        (ROOT / "Articles").glob("*.py")
    )
    # Inclui tabelas térmicas versionadas: também fazem parte do cálculo.
    paths += sorted((ROOT / "src/CosmoTransitions").glob("*.npz"))
    files = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "CosmoTransitions")
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "settings": asdict(settings),
        "grid": grid,
        "source_sha256": files,
        "versions": versions,
        "python": platform.python_version(),
        "model_constants": asdict(ModelParameters()),
    }
    result = {
        **identity,
        "fingerprint": hashlib.sha256(canonical_json(identity).encode()).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "resummation": "fixed_thesis_gauge_daisy",
        "alpha_for_gw": "signed_trace_anomaly",
        "frequency_unit_csv": "Hz",
        "frequency_unit_core": "mHz",
        "spectrum_defaults": {
            "v_w": settings.wall_velocity,
            "g_star": settings.g_star,
            "T_star": "Tn",
            "kappa_sw": "core_default",
            "y_sup_sw": "core_default",
            "epsilon_turb": 0.05,
            "kappa_coll": 0.0,
        },
        "PIS_reference": "https://arxiv.org/abs/2002.04615",
        "PIS_interpretation": "acoustic peak / published s-channel PIS fit; not total SNR",
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return json_safe(result)


@contextmanager
def single_writer(output):
    """Trava de sistema operacional, liberada inclusive após crash do processo."""
    with (output / ".writer.lock").open("a+b") as handle:
        handle.seek(0)
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "Já existe um coletor usando esta pasta de resultados."
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def open_database(output, manifest):
    connection = sqlite3.connect(output / "scan.sqlite")
    connection.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS points (
            point_id TEXT PRIMARY KEY, status TEXT NOT NULL, record TEXT NOT NULL,
            details_path TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS transitions (
            point_id TEXT NOT NULL, transition_index INTEGER NOT NULL, record TEXT NOT NULL,
            PRIMARY KEY (point_id, transition_index));
    """)
    old = connection.execute(
        "SELECT value FROM metadata WHERE key='manifest'"
    ).fetchone()
    if old and json.loads(old[0])["fingerprint"] != manifest["fingerprint"]:
        connection.close()
        raise ValueError(
            "Configuração, código ou versões mudaram. Use outra pasta --output."
        )
    with connection:
        connection.execute(
            "INSERT OR IGNORE INTO metadata VALUES ('manifest',?)",
            (canonical_json(manifest),),
        )
    if not old:
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return connection


def save_point(connection, output, result):
    """Detalhes atômicos primeiro; commit único do ponto e suas transições depois."""
    row, transitions, details = result
    key = row["point_id"]
    serialized = canonical_json(details)
    digest = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    # Content-addressed files prevent a crash during retry from overwriting
    # details still referenced by the preceding committed row.
    relative = Path("details") / key[:2] / f"{key}_{digest}.json.gz"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        stream.write(serialized)
    temporary.replace(destination)
    row["details_path"] = relative.as_posix()
    with connection:
        connection.execute("DELETE FROM transitions WHERE point_id=?", (key,))
        connection.execute(
            "INSERT OR REPLACE INTO points VALUES (?,?,?,?)",
            (key, row["status"], canonical_json(row), relative.as_posix()),
        )
        connection.executemany(
            "INSERT INTO transitions VALUES (?,?,?)",
            [(key, tr["transition_index"], canonical_json(tr)) for tr in transitions],
        )


def export_csv(connection, output):
    """Exportação em fluxo; não carrega a grade inteira na memória."""
    for table in ("points", "transitions"):
        keys = {"point_id", "status"}
        for (record,) in connection.execute(f"SELECT record FROM {table}"):
            keys.update(json.loads(record))
        order = "point_id" + (", transition_index" if table == "transitions" else "")
        temporary = output / f"{table}.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["point_id", "status"]
                + sorted(keys - {"point_id", "status"}),
            )
            writer.writeheader()
            for (record,) in connection.execute(
                f"SELECT record FROM {table} ORDER BY {order}"
            ):
                writer.writerow(json.loads(record))
        temporary.replace(output / f"{table}.csv")


def pending_points(connection, output, points, retry_failed=False):
    for params in points:
        old = connection.execute(
            "SELECT status, details_path FROM points WHERE point_id=?",
            (point_id(params),),
        ).fetchone()
        if (
            old is None
            or not (output / old[1]).exists()
            or (
                retry_failed
                and old[0] in ("numerical_failure", "observables_unresolved")
            )
        ):
            yield params


def run_scan(
    output, points, settings, manifest, workers=1, max_points=None, retry_failed=False
):
    output.mkdir(parents=True, exist_ok=True)
    with single_writer(output):
        connection = open_database(output, manifest)
        todo = itertools.islice(
            pending_points(connection, output, points, retry_failed), max_points
        )
        completed = 0

        def store_result(result):
            nonlocal completed
            save_point(connection, output, result)
            completed += 1
            row = result[0]
            print(
                f"[{completed}] m6={row['m6_GeV']:g}, m8={row['m8_GeV']:g}, "
                f"C={row['C']:g}: {row['status']} ({row['runtime_seconds']:.1f}s)",
                flush=True,
            )
            # SQLite já está atualizado. Exportar toda a tabela a cada lote
            # pequeno tornaria uma campanha grande quadraticamente custosa.

        try:
            if workers == 1:
                for params in todo:
                    store_result(evaluate_point(params, settings))
            else:
                # Fila limitada: um scan com centenas de milhares de pontos não
                # cria centenas de milhares de Futures. Spawn funciona no Windows.
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                ) as pool:
                    pending = {}
                    for _ in range(workers * 2):
                        params = next(todo, None)
                        if params is not None:
                            pending[pool.submit(evaluate_point, params, settings)] = (
                                params
                            )
                    while pending:
                        done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                        if not done:
                            print(
                                f"Calculando {len(pending)} pontos; {completed} salvos nesta execução...",
                                flush=True,
                            )
                        for future in done:
                            pending.pop(future)
                            store_result(future.result())
                            params = next(todo, None)
                            if params is not None:
                                pending[
                                    pool.submit(evaluate_point, params, settings)
                                ] = params
        finally:
            export_csv(connection, output)
            connection.close()
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "Articles/results/combined"
    )
    parser.add_argument(
        "--m6",
        nargs=3,
        type=float,
        default=(500, 2000, 5),
        metavar=("MIN", "MAX", "STEP"),
    )
    parser.add_argument(
        "--C",
        nargs=3,
        type=float,
        default=(0, 10, 0.02),
        metavar=("MIN", "MAX", "STEP"),
    )
    parser.add_argument("--m8", nargs="+", type=float, default=M8_SCENARIOS_GEV)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--action-tolerance", type=float, default=0.5)
    parser.add_argument("--beta-step", type=float, default=0.5)
    parser.add_argument("--beta-check", action="store_true")
    parser.add_argument("--T-min", type=float, default=1.0)
    parser.add_argument("--T-max", type=float, default=250.0)
    parser.add_argument("--phi-max", type=float, default=1000.0)
    parser.add_argument("--n-phi", type=int, default=1200)
    parser.add_argument("--n-T-seeds", type=int, default=5)
    args = parser.parse_args(argv)
    if args.export_only:
        database = args.output / "scan.sqlite"
        if not database.exists():
            parser.error(f"Banco inexistente: {database}")
        with single_writer(args.output), sqlite3.connect(database) as connection:
            export_csv(connection, args.output)
        return 0
    settings = Settings(
        T_min=args.T_min,
        T_max=args.T_max,
        phi_max=args.phi_max,
        n_phi=args.n_phi,
        n_T_seeds=args.n_T_seeds,
        action_tolerance=args.action_tolerance,
        beta_step=args.beta_step,
        beta_check=args.beta_check,
    )
    try:
        settings.validate()
        masses, couplings = inclusive_axis(*args.m6), inclusive_axis(*args.C)
        if masses[0] <= 0 or couplings[0] < 0 or args.workers < 1:
            raise ValueError("Exige m6>0, C>=0, workers>=1.")
        if args.max_points is not None and args.max_points < 1:
            raise ValueError("max-points deve ser positivo.")
        for m8 in args.m8:
            if math.isnan(m8) or m8 <= 0:
                raise ValueError("m8 deve ser positivo ou inf.")
        grid = {
            "m6_range_GeV": args.m6,
            "C_range": args.C,
            "m8_GeV": args.m8,
            "include_baselines": not args.no_baselines,
            "Lambda_GeV": 1000.0,
        }
        n_mass = len(masses) + int(not args.no_baselines)
        n_C = len(couplings) + int(not args.no_baselines and 0 not in couplings)
        n_m8 = len(set(args.m8) | ({math.inf} if not args.no_baselines else set()))
        print(
            f"Grade: {n_mass * n_C * n_m8:,} pontos; m6 em GeV, C adimensional; Lambda=1000 GeV."
        )
        print(
            f"Ressoma gauge fixa; |S3/T-140|<={settings.action_tolerance}; beta ordem 2, h={settings.beta_step} GeV."
        )
        if args.dry_run:
            return 0
        # Inicializa tabelas uma vez antes de criar processos (cache determinístico).
        from CosmoTransitions import Jb, Jf

        Jb(np.array([0.0]), approx="spline")
        Jf(np.array([0.0]), approx="spline")
        manifest = provenance(settings, grid)
        points = scan_points(masses, couplings, args.m8, not args.no_baselines)
        count = run_scan(
            args.output,
            points,
            settings,
            manifest,
            args.workers,
            args.max_points,
            args.retry_failed,
        )
        print(
            f"Concluído: {count} pontos novos/reprocessados. Dados em {args.output.resolve()}"
        )
        return 0
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
