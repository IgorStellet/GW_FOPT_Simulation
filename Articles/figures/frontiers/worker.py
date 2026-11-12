"""Worker novo: muda somente a escolha do passo, chama o recorder v4.

Não implementa outra ação, derivada, potencial ou termodinâmica. A procura
de fases/Tn é a mesma. O passo de beta permanece de ordem 2 e é comparado
com seu meio passo pelo código original, que continua reproduzível.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import traceback
from dataclasses import replace
from functools import partial
from pathlib import Path

from Articles.collect_data import (
    Settings,
    _evaluate_point_once,
    json_safe,
    transition_record,
)
from Articles.combined_model import ModelParameters


def margin_record(transition, phases, derivatives, settings, *, numerics):
    Tn = float(transition["Tnuc"])
    effective = settings.beta_step
    margin = None
    if transition["trantype"] == 1:
        low = max(
            float(phases[transition[key + "_phase"]].T[0]) for key in ("high", "low")
        )
        high = min(
            float(phases[transition[key + "_phase"]].T[-1]) for key in ("high", "low")
        )
        critical = transition.get("crit_trans") or {}
        if critical.get("Tcrit") is not None:
            high = min(high, float(critical["Tcrit"]))
        margin = min(Tn - low, high - Tn)
        # Se nem o piso cabe, o recorder original deixa beta inconclusivo.
        # Não extrapola fases nem troca o stencil central por um unilateral.
        if math.isfinite(margin) and margin > 0:
            effective = min(
                settings.beta_step,
                max(numerics["minimum_step_GeV"], numerics["margin_fraction"] * margin),
            )
    row, details = transition_record(
        transition, phases, derivatives, replace(settings, beta_step=effective)
    )
    row.update(
        beta_step_policy="metastable_margin_v1",
        beta_maximum_step_GeV=settings.beta_step,
        beta_metastable_margin_GeV=margin,
    )
    return row, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    params = ModelParameters(
        **{
            key: math.inf if value == "inf" else value
            for key, value in request["params"].items()
        }
    )
    settings = Settings(**request["settings"])
    numerics = request["numerics"]
    recorder = partial(margin_record, numerics=numerics)
    try:
        result = _evaluate_point_once(params, settings, record_transition=recorder)
    except Exception as error:  # noqa: BLE001 - falha vira registro, não exclusão
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
        json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
