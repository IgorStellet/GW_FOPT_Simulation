"""Avaliações dirigidas: mesmos potenciais e observáveis, recuperação auditável."""

from __future__ import annotations

import argparse
import json
import math
import signal
import traceback
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from Articles.adaptive_scan import nucleation_decision
from Articles.collect_data import Settings, _evaluate_point_once, json_safe
from Articles.combined_model import ModelParameters
from Articles.figures.data import critical_ratio
from Articles.figures.frontiers.data import alpha_valid, beta_valid
from Articles.figures.frontiers.worker import margin_record


def ladder_record(transition, phases, derivatives, settings, *, numerics):
    """Ordem 2 em cada avaliação e o mesmo teste h/2 do recorder original.

    A ordem dos passos é fixada no plano, sem consultar SNR. Primeiro se usa
    0.5 GeV; 1 GeV testa cancelamento numérico; 0.25 GeV testa truncamento.
    O limite de metastabilidade permanece ativo e passos repetidos são pulados.
    Todas as tentativas, inclusive as rejeitadas, ficam no arquivo de detalhes.
    """
    attempts, tested = [], set()
    chosen = None
    margin = None
    for maximum in numerics["maximum_steps_GeV"]:
        effective = maximum
        if margin is not None and math.isfinite(margin) and margin > 0:
            effective = min(
                maximum,
                max(numerics["minimum_step_GeV"], numerics["margin_fraction"] * margin),
            )
        if any(
            math.isclose(effective, h, rel_tol=1e-12, abs_tol=1e-12) for h in tested
        ):
            continue
        row, extra = margin_record(
            transition,
            phases,
            derivatives,
            replace(settings, beta_step=maximum, beta_check=True),
            numerics=numerics,
        )
        tested.add(row.get("beta_step_GeV", effective))
        margin = row.get("beta_metastable_margin_GeV")
        attempts.append(
            {
                "maximum_step_GeV": maximum,
                "observables": dict(row),
                "diagnostics": extra,
            }
        )
        accepted = alpha_valid(row) and beta_valid(row)
        if chosen is None or accepted:
            chosen = row, extra
        if accepted:
            break
        flags = str(row.get("quality_flags", ""))
        if "beta" not in flags:
            break  # Uma falha de nucleação/termodinâmica exige outro diagnóstico.
    row, extra = chosen
    extra = {**extra, "beta_step_ladder": attempts}
    row.update(
        beta_step_policy="central_step_ladder_v1", beta_step_attempts=len(attempts)
    )
    return row, extra


def evaluate_coverage(params, settings, numerics):
    recorder = partial(ladder_record, numerics=numerics)
    first = _evaluate_point_once(params, settings, record_transition=recorder)
    result = first
    # Refinamento de fases determinístico, com os mesmos ajustes utilizados na
    # recuperação v4. Mantém o critério S3/T e o potencial; muda a resolução.
    if numerics.get("phase_recovery") and (
        nucleation_decision(first[0], first[1]) != "yes"
        or any(
            t.get("nucleation_accepted") and critical_ratio(t, first[2])[0] is None
            for t in first[1]
        )
    ):
        refined = replace(
            settings,
            n_phi=settings.n_phi * 2,
            n_T_seeds=2 * settings.n_T_seeds - 1,
            deltaX_target=settings.deltaX_target / 2,
            phase_dtstart_frac=settings.phase_dtstart_frac / 5,
            phase_tjump_frac=settings.phase_tjump_frac / 5,
            minima_phitol=settings.minima_phitol / 10,
            root_T_tolerance=settings.root_T_tolerance / 10,
        )
        try:
            recovered = _evaluate_point_once(
                params, refined, record_transition=recorder
            )
        except Exception as error:  # noqa: BLE001 - preserva a tentativa física inicial.
            first[2]["phase_recovery_error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "settings": asdict(refined),
            }
            first[0]["evaluation_attempts"] = 2
            recovered = None
        # Observações válidas anteriores continuam disponíveis no conjunto
        # original. A tentativa nova guarda também o diagnóstico completo.
        if recovered is not None:
            recovered[2]["phase_recovery"] = {
                "settings": asdict(refined),
                "initial_point": first[0],
                "initial_details": first[2],
            }
            result = recovered
    row, transitions, details = result
    for transition in transitions:
        ratio, source = critical_ratio(transition, details)
        transition.update(vev_over_Tc=ratio, critical_ratio_source=source)
        if ratio is not None:
            transition["phi_low_Tc_GeV"] = ratio * transition["Tc_GeV"]
    row.update(
        calculation_mode="coverage_observables_v2",
        evaluation_attempts=max(
            row.get("evaluation_attempts", 1), 2 if result is not first else 1
        ),
    )
    return row, transitions, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    params = ModelParameters(
        **{k: math.inf if v == "inf" else v for k, v in request["params"].items()}
    )
    try:
        result = evaluate_coverage(
            params, Settings(**request["settings"]), request["numerics"]
        )
    except Exception as error:  # noqa: BLE001 - falha física vira dado por ponto.
        result = (
            {
                **request["params"],
                "status": "numerical_failure",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            [],
            {"traceback": traceback.format_exc()},
        )
    temporary = args.result.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(json_safe(result), allow_nan=False), encoding="utf-8"
    )
    temporary.replace(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
