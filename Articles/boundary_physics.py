"""Registra nucleação/VEVs sem calcular termodinâmica, beta ou espectros."""

from __future__ import annotations

import argparse
import json
import math
import signal
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from Articles.collect_data import (
    Settings,
    _evaluate_point_once,
    evaluate_point,
    json_safe,
)
from Articles.combined_model import ModelParameters


def boundary_record(transition, phases, derivatives, settings):
    """Os VEVs são os mínimos já refinados pelo tunneling do núcleo em Tn.

    Não usa valores em Tc, extrapolação de fases, nem calcula novos bounces.
    O valor de campo da fase final e o salto entre fases ficam separados.
    """
    Tn = float(transition["Tnuc"])
    row = {"status": "not_first_order", "trantype": int(transition["trantype"]),
           "Tn_GeV": Tn, "calculation_mode": "nucleation_only"}
    extra = {"quality_flags": [], "action_samples": []}
    if row["trantype"] != 1:
        return row, extra
    action = float(transition["action"])
    row.update(S3_GeV=action, S3_over_T=action / Tn,
               nucleation_residual=action / Tn - settings.nucleation_target)
    row["nucleation_accepted"] = (Tn > 0 and math.isfinite(action) and action >= 0
                                 and abs(row["nucleation_residual"]) <= settings.action_tolerance)
    critical = transition.get("crit_trans") or {}
    row["Tc_GeV"] = critical.get("Tcrit")
    try:
        high = float(np.asarray(transition["high_vev"]).ravel()[0])
        low = float(np.asarray(transition["low_vev"]).ravel()[0])
        if not (Tn > 0 and all(math.isfinite(v) for v in (Tn, high, low))):
            raise ValueError("VEVs/Tn não finitos ou Tn não positivo")
        row.update(phi_high_GeV=high, phi_low_GeV=low,
                   vev_over_Tn=abs(low) / Tn, delta_phi_over_Tn=abs(low - high) / Tn)
        row["strong_by_field_ratio"] = row["delta_phi_over_Tn"] >= settings.strong_threshold
    except (KeyError, ValueError, TypeError) as error:
        extra["quality_flags"].append(f"field_ratio_unresolved: {error}")
    if not row["nucleation_accepted"]:
        extra["quality_flags"].append("nucleation_residual_outside_tolerance")
    row["quality_flags"] = "; ".join(extra["quality_flags"])
    row["status"] = "observables_unresolved" if extra["quality_flags"] else "nucleated"
    return row, extra


def classify(row, transitions, criterion="strong", threshold=1.0):
    """Classifica a REGIÃO SELECIONADA, sem transformar fraca em não nucleada.

    Uma transição forte aceita basta para incluir o ponto. Para excluir por
    força, todas as candidatas de primeira ordem precisam estar resolvidas.
    Avisos relativos apenas a beta/GW de dados antigos não apagam Tn/VEVs.
    """
    first_order = [t for t in transitions if t.get("trantype") == 1]
    accepted = [t for t in first_order if t.get("nucleation_accepted")]
    if accepted:
        if criterion == "nucleation":
            return "yes", "accepted_nucleation"
        ratios = []
        for tr in accepted:
            ratio = tr.get("vev_over_Tn")
            if ratio is None and tr.get("phi_low_GeV") is not None and tr.get("Tn_GeV", 0) > 0:
                ratio = abs(tr["phi_low_GeV"]) / tr["Tn_GeV"]
            ratios.append(ratio)
        if any(isinstance(v, (int, float)) and math.isfinite(v) and v > threshold for v in ratios):
            return "yes", "accepted_strong_nucleation"
        if (len(accepted) != len(first_order)
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in ratios)):
            return "unknown", "strength_unresolved"
        return "no", "weak_nucleation"
    if row["status"] in ("no_first_order_found", "no_nucleation_found") and not first_order:
        return "no", "no_nucleation_found"
    return "unknown", row.get("error_type", row["status"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    # O coordenador trata Ctrl+C e deixa o ponto terminar dentro do orçamento.
    # Ignorar aqui evita perder o ponto por um sinal de console em grupo.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    params = ModelParameters(**{k: math.inf if v == "inf" else v for k, v in request["params"].items()})
    settings = Settings(**request["settings"])
    try:
        if request["mode"] == "boundaries":
            result = _evaluate_point_once(params, settings, record_transition=boundary_record)
        else:
            result = evaluate_point(params, settings)
        result[0]["calculation_mode"] = request["mode"]
    except Exception as error:  # noqa: BLE001 - o coordenador salva a falha
        result = ({**asdict(params), "status": "numerical_failure",
                   "error_type": type(error).__name__, "error_message": str(error)},
                  [], {"traceback": traceback.format_exc()})
    temporary = args.result.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
