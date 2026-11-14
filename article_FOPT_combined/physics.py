"""Avaliação de um ponto: primeiro nucleação; depois termodinâmica e GW.

Não contém integrais térmicas, integradores de bounce ou um rastreador de fases
próprios. Esses cálculos pertencem ao src. Este arquivo transforma suas saídas
em registros físicos, com critérios explícitos e dados para auditar beta.
"""

from __future__ import annotations

import argparse
import io
import math
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace

import numpy as np

from . import core
from .config import Numerics
from .model import Potential
from .storage import atomic_json, parameters_from_dict, read_json, safe


class NucleationCriterion:
    """Aceita |S3/T-140|<=0.5, sem exigir precisão artificial da raiz em T.

    root_T_tolerance continua sendo uma salvaguarda do resolvedor; a aceitação
    física depende do residual da ação. Todas as ações consultadas são salvas.
    """

    def __init__(self, target, tolerance):
        self.target, self.tolerance, self.samples = target, tolerance, {}

    def __call__(self, action, temperature):
        residual = action / temperature - self.target
        self.samples[float(temperature)] = {
            "T_GeV": temperature,
            "S3_GeV": action,
            "S3_over_T": action / temperature,
            "residual": residual,
        }
        if math.isfinite(action) and action < 0:
            raise ValueError("Ação de nucleação negativa.")
        return 0.0 if abs(residual) <= self.tolerance else residual


class BoundedLog(io.StringIO):
    """Limita o texto de diagnóstico de cada ponto a 32 kB."""

    def write(self, text):
        remaining = max(0, 32_000 - self.tell())
        super().write(text[:remaining])
        return len(text)


def derivatives_for(model, settings):
    """Derivadas do potencial completo, incluindo a dependência térmica dos anéis."""
    return core.generic.build_finite_T_derivatives(
        model.Vtot,
        Ndim=1,
        x_eps=settings.x_eps,
        T_eps=settings.T_eps,
        deriv_order=4,
        X_ref=np.array([0.0]),
    )


def build_history(model, derivatives, settings, criterion):
    """Rastreia fases e nucleação pelo pipeline do src, na janela declarada."""
    phi_max = min(settings.phi_max, 0.98 * model.domain_limit)
    if phi_max <= model.v:
        raise ValueError("A janela de campos deve incluir o vácuo eletrofraco.")

    def forbidden(X):
        phi = float(np.asarray(X).ravel()[0])
        return phi < -10 * settings.x_eps or phi >= phi_max

    with core.ordered_phase_inputs():
        summary = core.generic._build_phases_and_transitions(
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
            deltaX_target=settings.phase_delta,
            dtstart_frac=settings.phase_step,
            tjump_frac=settings.phase_step,
            forbidCrit=forbidden,
            nuclCriterion=criterion,
            Tn_Ttol=settings.root_T_tolerance,
            Tn_maxiter=100,
            tunnelFromPhase_args={
                "phitol": settings.minima_tolerance,
                "overlapAngle": 0.0,
                "verbose": False,
            },
            verbose=False,
        )
    return summary


def initial_phase_check(model, settings):
    """Verifica a fase inicial antes de gastar tempo rastreando uma história inadequada.

    Um mínimo de campo alto pode refletir o limite das massas hard de alta T.
    A checagem descreve o potencial aproximado; não prova não restauração física.
    """
    phi_max = min(settings.phi_max, 0.98 * model.domain_limit)
    grid = np.linspace(0, phi_max, settings.n_phi)
    values = model(grid, settings.T_max)
    reference = float(values[0])
    index = int(np.nanargmin(values))
    difference = float(values[index] - reference)
    tolerance = 1e-8 * max(abs(reference), 1)
    return {
        "T_GeV": settings.T_max,
        "grid_min_phi_GeV": float(grid[index]),
        "grid_min_minus_origin_GeV4": difference,
        "symmetric_on_grid": difference >= -tolerance,
        "phi_search_max_GeV": phi_max,
    }


def pack_history(history):
    """Salva fases e transições leves para reutilizar as bordas na segunda etapa."""
    packed = []
    for transition in history["full_transitions"]:
        row = {
            k: transition[k]
            for k in ("Tnuc", "action", "trantype", "high_vev", "low_vev", "crit_trans")
            if k in transition
        }
        row.update(high_phase=str(transition["high_phase"]), low_phase=str(transition["low_phase"]))
        packed.append(row)
    return safe(
        {
            "phases": {
                str(key): {"T": p.T, "X": p.X, "dXdT": p.dXdT}
                for key, p in history["phases"].items()
            },
            "full_transitions": packed,
            "critical_transitions": history["critical_transitions"],
            "start_phase_key": str(history["start_phase_key"]),
        }
    )


def unpack_history(context):
    """Recupera a história exata de um ponto de borda, sem procurar Tn novamente."""
    return {
        **context,
        "phases": {key: core.phase_from_record(key, row) for key, row in context["phases"].items()},
    }


def nucleation_record(transition, index, settings):
    """Extrai ação, campos, Tc e força, sem calcular alpha ou derivadas de ações."""
    Tn = float(transition["Tnuc"])
    row = {
        "transition_index": index,
        "trantype": int(transition["trantype"]),
        "Tn_GeV": Tn,
        "nucleation_accepted": False,
        "high_phase": str(transition["high_phase"]),
        "low_phase": str(transition["low_phase"]),
    }
    critical = transition.get("crit_trans") or {}
    if critical.get("Tcrit") is not None:
        row["Tc_GeV"] = float(critical["Tcrit"])
        for side in ("high", "low"):
            if f"{side}_vev" in critical:
                row[f"phi_{side}_Tc_GeV"] = float(np.asarray(critical[f"{side}_vev"]).ravel()[0])
        if row["Tc_GeV"] > 0 and "phi_low_Tc_GeV" in row:
            row["vev_over_Tc"] = abs(row["phi_low_Tc_GeV"]) / row["Tc_GeV"]
            row["supercooling_fraction"] = 1 - Tn / row["Tc_GeV"]
    if row["trantype"] != 1:
        return row
    action = float(transition["action"])
    row.update(
        S3_GeV=action,
        S3_over_T=action / Tn,
        nucleation_residual=action / Tn - settings.nucleation_target,
    )
    row["nucleation_accepted"] = (
        Tn > 0
        and math.isfinite(action)
        and action >= 0
        and abs(row["nucleation_residual"]) <= settings.action_tolerance
    )
    for side in ("high", "low"):
        row[f"phi_{side}_GeV"] = float(np.asarray(transition[f"{side}_vev"]).ravel()[0])
    row["vev_over_Tn"] = abs(row["phi_low_GeV"]) / Tn
    row["delta_phi_over_Tn"] = abs(row["phi_low_GeV"] - row["phi_high_GeV"]) / Tn
    row["selected_strong"] = (
        row["nucleation_accepted"] and row["vev_over_Tn"] > settings.strong_threshold
    )
    return row


def beta_record(calc, row, settings, lower, upper):
    """Escolhe um passo admissível e verifica estabilidade até convergir ou atingir o piso.

    A primeira estimativa usa o máximo solicitado quando ele cabe. Perto de
    extremos, começa em 80% da maior abertura admissível. No modo adaptativo,
    compara h e h/2 também para os pontos que comportam o stencil solicitado.
    Conserva a estimativa mais fina do par estável dentro da tolerância
    aceito e registra todas as tentativas. Falha permanece um dado pendente.
    """
    requested = settings.beta_step_GeV
    Tn = row["Tn_GeV"]
    maximum = max(min(Tn - lower, upper - Tn), (Tn - lower) / 2, (upper - Tn) / 2)
    adapted = requested >= maximum
    if adapted and not settings.adaptive_beta:
        raise ValueError("Nenhum stencil cabe com o passo fixo solicitado.")
    step = min(requested, 0.8 * maximum) if adapted else requested
    if step <= 0 or (adapted and step / 2 < settings.beta_min_step_GeV):
        raise ValueError(
            "Intervalo de coexistência insuficiente para validar beta acima do passo mínimo."
        )
    beta, stencil, samples = core.beta_at_transition(calc, row, step, lower, upper)
    result = {
        "beta_over_H": beta,
        "beta_stencil": stencil,
        "beta_stencil_points": samples,
        "beta_step_requested_GeV": requested,
        "beta_step_GeV": step,
        "beta_step_adapted": adapted,
        "beta_stability_checked": settings.adaptive_beta,
        "beta_stability_passed": not settings.adaptive_beta,
    }
    if settings.adaptive_beta:
        history = [{"step_GeV": step, "beta_over_H": beta, "stencil": stencil, "points": samples}]
        while step / 2 >= settings.beta_min_step_GeV:
            try:
                check, check_stencil, check_samples = core.beta_at_transition(
                    calc, row, step / 2, lower, upper
                )
            except Exception as error:
                result["beta_validation_error"] = f"{type(error).__name__}: {error}"
                break
            change = abs(check - beta) / max(abs(check), abs(beta), 1e-30)
            history.append(
                {
                    "step_GeV": step / 2,
                    "beta_over_H": check,
                    "stencil": check_stencil,
                    "points": check_samples,
                }
            )
            result.update(
                beta_over_H=check,
                beta_step_GeV=step / 2,
                beta_stencil=check_stencil,
                beta_stencil_points=check_samples,
                beta_check_step_GeV=step,
                beta_check_value=beta,
                beta_check_stencil=stencil,
                beta_check_points=samples,
                beta_relative_change=change,
                beta_stability_passed=change <= settings.beta_relative_tolerance,
                beta_step_adapted=True,
            )
            if result["beta_stability_passed"]:
                break
            step, beta, stencil, samples = step / 2, check, check_stencil, check_samples
        result["beta_convergence_history"] = history
    return result


def thermodynamic_record(row, phases, derivatives, model, settings):
    """Enriquece uma transição aceita, preservando alpha mesmo se beta falhar.

    As diferenças são sempre falso menos verdadeiro. Alpha_trace usa
    DeltaV-T*Delta(dV/dT)/4; alpha_energy usa DeltaV-T*Delta(dV/dT).
    Nenhum valor absoluto esconde um sinal termodinâmico inesperado.
    """
    result = dict(
        row,
        g_star=settings.g_star,
        v_w=settings.wall_velocity,
        beta_order=2,
        beta_step_GeV=settings.beta_step_GeV,
        observables_complete=False,
    )
    issues = []
    try:
        calc = core.calculator(
            derivatives, phases, row["high_phase"], row["low_phase"], settings.minima_tolerance
        )
    except Exception as error:
        result["quality_flags"] = [f"phase_overlap: {type(error).__name__}: {error}"]
        return result
    Tn = row["Tn_GeV"]
    try:
        high, low = calc.high_phase.valAt(Tn), calc.low_phase.valAt(Tn)
        Vh, Vl = float(derivatives.V(high, Tn)), float(derivatives.V(low, Tn))
        dTh, dTl = float(derivatives.dV_dT(high, Tn)), float(derivatives.dV_dT(low, Tn))
        delta, derivative = Vh - Vl, dTh - dTl
        radiation = np.pi**2 * settings.g_star * Tn**4 / 30
        result.update(
            V_high_GeV4=Vh,
            V_low_GeV4=Vl,
            dVdT_high_GeV3=dTh,
            dVdT_low_GeV3=dTl,
            delta_V_GeV4=delta,
            delta_dVdT_GeV3=derivative,
            delta_rho_GeV4=delta - Tn * derivative,
            delta_theta_GeV4=delta - Tn * derivative / 4,
            rho_rad_GeV4=radiation,
            alpha_energy=(delta - Tn * derivative) / radiation,
            alpha_trace=(delta - Tn * derivative / 4) / radiation,
        )
        ph, pg = model.scalar_self_energies(float(low[0]), Tn)
        bare = model.masses_squared(float(low[0]))
        result.update(
            screened_h_at_low_GeV2=float(bare["h"] + ph),
            screened_G_at_low_GeV2=float(bare["G"] + pg),
        )
    except Exception as error:
        issues.append(f"thermodynamics: {type(error).__name__}: {error}")
    try:
        lower = max(float(calc.high_phase.T[0]), float(calc.low_phase.T[0]))
        upper = min(
            float(calc.high_phase.T[-1]), float(calc.low_phase.T[-1]), row.get("Tc_GeV", math.inf)
        )
        result.update(phase_T_min_GeV=lower, phase_T_max_GeV=upper)
        result.update(beta_record(calc, row, settings, lower, upper))
        if not result["beta_stability_passed"]:
            issues.append("beta_step_sensitivity: passo adaptado sem estabilidade suficiente")
    except Exception as error:
        issues.append(f"beta: {type(error).__name__}: {error}")
    valid = all(
        isinstance(result.get(k), (int, float)) and math.isfinite(result[k]) and result[k] > 0
        for k in ("alpha_trace", "beta_over_H", "Tc_GeV", "Tn_GeV")
    )
    if valid:
        peak_mHz = core.gw.gw_f_sw_peak(
            result["beta_over_H"], Tn, settings.g_star, settings.wall_velocity
        )
        amplitude = float(
            np.asarray(core.sound_spectrum(np.array([peak_mHz * 1e-3]), result)).ravel()[0]
        )
        result.update(f_sw_peak_Hz=float(peak_mHz * 1e-3), omega_sw_peak_h2=amplitude)
        for name, sensitivity in (
            ("LISA", core.gw.lisa_sensitivity_s_pis),
            ("DECIGO", core.gw.decigo_sensitivity_s_pis),
            ("BBO", core.gw.bbo_sensitivity_s_pis),
        ):
            value = float(np.asarray(sensitivity(np.array([peak_mHz]))).ravel()[0])
            result[f"{name}_PIS_h2_1yr_SNR1"] = value
            result[f"{name}_PIS_ratio_1yr_SNR1"] = amplitude / value
        result["observables_complete"] = not issues
    else:
        issues.append("Alpha/beta/Tc/Tn positivos e finitos são necessários para o espectro.")
    result["quality_flags"] = issues
    result["beta_action_samples"] = [
        {"T_GeV": T, "S3_GeV": action, "S3_over_T": action / T}
        for T, action in sorted(calc._S3_cache.items())
    ]
    return result


def classify(rows, critical_count):
    """Uma transição forte aceita seleciona o ponto; falha permanece unknown."""
    if any(row.get("selected_strong") for row in rows):
        return "yes", "strong_nucleation"
    first = [r for r in rows if r["trantype"] == 1]
    if any(not r.get("nucleation_accepted") for r in first):
        return "unknown", "action_residual_unresolved"
    if first:
        return "no", "weak_nucleation"
    return "no", "no_nucleation_found" if critical_count else "no_first_order_found"


def evaluate_once(parameters, settings, mode="nucleation", cached=None):
    """Uma tentativa física; todo erro é um registro explícito, nunca uma exclusão."""
    started = time.monotonic()
    output = {
        "parameters": asdict(parameters),
        "settings": asdict(settings),
        "transitions": [],
        "summary": {"status": "numerical_failure", "decision": "unknown"},
        "context": {},
    }
    criterion = NucleationCriterion(settings.nucleation_target, settings.action_tolerance)
    log = BoundedLog()
    with redirect_stdout(log), redirect_stderr(log), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        try:
            model = Potential(parameters)
            output["diagnostics"] = model.diagnostics()
            if not output["diagnostics"]["ew_below_origin"]:
                output["summary"] = {
                    "status": "ew_vacuum_unfavoured",
                    "decision": "no",
                    "reason": "V(v,0)>=V(0,0)",
                }
            else:
                initial = initial_phase_check(model, settings)
                output["diagnostics"]["initial_phase"] = initial
                if not initial["symmetric_on_grid"]:
                    output["summary"] = {
                        "status": "initial_phase_unresolved",
                        "decision": "unknown",
                        "reason": "high_temperature_phase_not_symmetric",
                    }
                    output["runtime_seconds"] = time.monotonic() - started
                    return safe(output)
                derivatives = derivatives_for(model, settings)
                if cached and cached.get("context", {}).get("phases"):
                    history = unpack_history(cached["context"])
                    output["reused_boundary_history"] = True
                else:
                    history = build_history(model, derivatives, settings, criterion)
                    history = unpack_history(pack_history(history))
                output["context"] = pack_history(history)
                rows = [
                    nucleation_record(t, i, settings)
                    for i, t in enumerate(history["full_transitions"])
                ]
                if mode == "observables":
                    rows = [
                        thermodynamic_record(r, history["phases"], derivatives, model, settings)
                        if r.get("nucleation_accepted")
                        else r
                        for r in rows
                    ]
                output["transitions"] = rows
                ncrit = sum(t["trantype"] == 1 for t in history["critical_transitions"])
                decision, reason = classify(rows, ncrit)
                # A história cosmológica começa na fase simétrica dentro da janela.
                start = history["phases"][str(history["start_phase_key"])]
                start_phi = float(np.asarray(start.valAt(settings.T_max)).ravel()[0])
                if abs(start_phi) / settings.T_max > 0.1:
                    decision, reason = "unknown", "high_temperature_phase_not_symmetric"
                selected = next((r for r in rows if r.get("selected_strong")), None)
                accepted = selected or next((r for r in rows if r.get("nucleation_accepted")), None)
                status = "nucleated" if accepted else reason
                if mode == "observables" and selected and not selected.get("observables_complete"):
                    status = "observables_unresolved"
                output["summary"] = {
                    **(accepted or {}),
                    "status": status,
                    "decision": decision,
                    "reason": reason,
                    "n_phases": len(history["phases"]),
                    "n_critical_first_order": ncrit,
                    "start_phi_GeV": start_phi,
                }
        except Exception as error:
            output.update(
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc(),
            )
            output["summary"].update(
                status="numerical_failure", decision="unknown", reason=type(error).__name__
            )
    output.update(
        nucleation_samples=list(criterion.samples.values()),
        log=log.getvalue(),
        warnings=sorted({str(w.message) for w in caught}),
        runtime_seconds=time.monotonic() - started,
    )
    return safe(output)


def evaluate(parameters, settings, mode="nucleation", cached=None):
    """Até duas tentativas; refina fases mantendo potencial e prescrições numéricas."""
    settings.validate()
    if mode not in ("nucleation", "observables"):
        raise ValueError("Modo físico desconhecido.")
    first = evaluate_once(parameters, settings, mode, cached)
    unresolved = (
        first["summary"]["decision"] == "unknown"
        or first["summary"]["status"] == "observables_unresolved"
    )
    if first["summary"].get("reason") == "high_temperature_phase_not_symmetric":
        return first
    if not unresolved or not settings.retry_resolution:
        return first
    refined = replace(
        settings,
        n_phi=2 * settings.n_phi,
        n_T_seeds=2 * settings.n_T_seeds - 1,
        phase_delta=settings.phase_delta / 2,
        phase_step=settings.phase_step / 5,
        minima_tolerance=settings.minima_tolerance / 10,
        retry_resolution=False,
    )
    second = evaluate_once(parameters, refined, mode)

    def score(record):
        summary = record["summary"]
        return (
            summary.get("observables_complete", False),
            summary["decision"] != "unknown",
            sum(summary.get(k) is not None for k in ("Tc_GeV", "alpha_trace", "beta_over_H")),
        )

    best = second if score(second) >= score(first) else first
    best = dict(best)
    best["resolution_attempts"] = [first, second]
    best["runtime_seconds"] = first["runtime_seconds"] + second["runtime_seconds"]
    return best


def main(argv=None):
    """Entrada privada do processo isolado; os comandos de usuário estão em __main__.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    request = read_json(args.request)
    result = evaluate(
        parameters_from_dict(request["parameters"]),
        Numerics(**request["settings"]),
        request["mode"],
        request.get("cached"),
    )
    atomic_json(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
