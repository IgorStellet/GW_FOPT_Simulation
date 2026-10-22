"""Alcance do LISA por cortes limitados em C, sem hipótese de monotonicidade.

Uma malha declarada procura os intervalos. Mudanças de classificação são
refinadas por bisseção; uma falha é unknown e interrompe esse bracket. A
malha não exclui ilhas mais estreitas que seu passo nem regiões fora do
envelope de C previamente selecionado.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

from Articles.boundary_scan import reuse_compatible, source_manifest
from Articles.collect_data import M8_SCENARIOS_GEV, point_id, validate_campaign
from Articles.combined_model import ModelParameters
from Articles.figures.data import campaign_folders, detector_metrics
from Articles.figures.frontiers.data import merge_intervals
from Articles.figures.frontiers.execution import (
    checkpoint,
    execution_manifest,
    live_store,
)
from Articles.figures.frontiers.planning import partition, separate_output


def summarize_cut(cut, samples, tolerance, maximum_gap=None):
    """Só preenche intervalos entre testemunhas detectáveis consecutivas."""
    samples = sorted(samples, key=lambda item: item["C"])
    intervals = [
        [sample["C"], sample["C"]] for sample in samples if sample["decision"] == "yes"
    ]
    brackets = []
    for left, right in pairwise(samples):
        a, b = left["decision"], right["decision"]
        if a == b == "yes" and (
            maximum_gap is None or right["C"] - left["C"] <= maximum_gap + 1e-10
        ):
            intervals.append([left["C"], right["C"]])
        elif {a, b} == {"yes", "no"}:
            inside, outside = (left, right) if a == "yes" else (right, left)
            brackets.append(
                {
                    "C_min": left["C"],
                    "C_max": right["C"],
                    "inside_C": inside["C"],
                    "outside_C": outside["C"],
                    "state": "bracketed"
                    if right["C"] - left["C"] <= tolerance + 1e-10
                    else "unresolved",
                    "kind": "SNR_threshold"
                    if outside.get("rho_sw") is not None
                    else "accepted_nucleation_boundary",
                }
            )
    return {
        **cut,
        "samples": samples,
        "intervals": merge_intervals(intervals),
        "brackets": brackets,
        "state": "complete_with_unresolved"
        if any(item["decision"] == "unknown" for item in samples)
        or any(item["state"] == "unresolved" for item in brackets)
        else "complete",
        "lower_domain_limited": bool(samples and samples[0]["decision"] == "yes"),
        "upper_domain_limited": bool(samples and samples[-1]["decision"] == "yes"),
    }


def trace_cut(cut, evaluate, tolerance):
    """Procura TODOS os brackets visíveis na malha, inclusive não monótonos."""
    answers = {}

    def sample(coupling):
        coupling = round(float(coupling), 12)
        if coupling not in answers:
            answers[coupling] = {**evaluate(coupling), "C": coupling}

    for coupling in cut["grid_C"]:
        sample(coupling)
    while True:
        ordered = sorted(answers)
        pending = [
            (low, high)
            for low, high in pairwise(ordered)
            if high - low > tolerance + 1e-10
            and {answers[low]["decision"], answers[high]["decision"]} == {"yes", "no"}
        ]
        if not pending:
            break
        for low, high in pending:
            sample((low + high) / 2)
    return summarize_cut(cut, list(answers.values()), tolerance)


def observation(store, mass, coupling, years, threshold):
    """Uma falha de beta não é SNR=0 nem ausência de transição."""
    store.evaluate(coupling, mass)
    key = point_id(ModelParameters(C=coupling, m6_GeV=mass, m8_GeV=store.plan.m8_GeV))
    point, transitions = store.cache[key]
    accepted = [
        item
        for item in transitions
        if item.get("nucleation_accepted") and item.get("trantype") == 1
    ]
    primary = min(
        accepted, key=lambda item: item.get("transition_index", 0), default={}
    )
    metrics = (
        detector_metrics(primary, store.manifest.get("spectrum_defaults", {}), years)
        if primary
        else {}
    )
    common = {
        "point_id": key,
        "point_status": point["status"],
        "strong_decision": point["boundary_decision"],
    }
    if metrics:
        rho = metrics["LISA_rho_sw"]
        return {
            **common,
            "decision": "yes" if rho >= threshold else "no",
            "rho_sw": rho,
            "reason": "resolved_acoustic_SNR",
            "f_sw_peak_Hz": metrics["f_sw_peak_Hz"],
            "omega_sw_peak_h2": metrics["omega_sw_peak_h2"],
        }
    no_signal = not any(item.get("trantype") == 1 for item in transitions) and point[
        "status"
    ] in ("no_first_order_found", "no_nucleation_found")
    return {
        **common,
        "decision": "no" if no_signal else "unknown",
        "rho_sw": None,
        "reason": "no_accepted_nucleation_found"
        if no_signal
        else point.get("error_type") or "GW_observables_unresolved",
    }


def run_lisa_shard(args):
    content = args.plan.read_bytes()
    plan = json.loads(content)
    if plan.get("schema") != "article_lisa_cuts_v1":
        raise ValueError("Use o plano criado por plan-lisa.")
    separate_output(args.output, plan)
    scenario, part, cuts = partition(
        plan["cuts"], args.task_index, args.shards_per_scenario
    )
    case = M8_SCENARIOS_GEV[scenario]
    settings, manifest = execution_manifest(
        plan, content, args, case, "article_lisa_cuts_v1"
    )
    output = args.output / f"m8-{scenario}-part-{part}"
    validate_campaign(output, manifest)
    for name in ("C_step", "C_tolerance", "observation_years", "snr_threshold"):
        if not math.isfinite(plan[name]) or plan[name] <= 0:
            raise ValueError(f"{name} inválido no plano.")
    if args.dry_run or not cuts:
        print(
            f"Tarefa {args.task_index}: {len(cuts)} cortes; configuração compatível; nenhum cálculo/pasta nova."
        )
        return 0
    report = {
        "state": "running",
        "kind": "lisa_cuts",
        "plan_sha256": hashlib.sha256(content).hexdigest(),
        "cuts_planned": len(cuts),
        "cuts": [],
        "unknown_points": [],
        "C_step": plan["C_step"],
        "C_tolerance": plan["C_tolerance"],
        "observation_years": plan["observation_years"],
        "snr_threshold": plan["snr_threshold"],
        "channel": "sound_waves_only_PIS",
    }
    with live_store(args, output, settings, manifest, case, report) as store:
        requested = {
            (coupling, float(cut["m6_GeV"]))
            for cut in cuts
            for coupling in cut["grid_C"]
        }
        for source in plan.get("reuse_sources", []):
            folder = Path(source["folder"])
            try:
                reuse_compatible(source_manifest(folder), manifest, "interior")
            except ValueError as error:
                print(
                    f"Sem reuso GW automático dessa origem: {folder}: {error}",
                    flush=True,
                )
                continue
            store.allowed_coordinates = {
                (item["C"], float(item["m6_GeV"])) for item in source["coordinates"]
            } & requested
            store.import_campaign(folder)
        store.allowed_coordinates = requested
        for root in args.reuse:
            for folder in campaign_folders(root):
                store.import_campaign(folder)
        store.allowed_coordinates = None
        for cut in cuts:
            partial = {**cut, "state": "running", "samples": []}
            report["cuts"].append(partial)

            def evaluate(coupling, cut=cut, partial=partial):
                result = observation(
                    store,
                    float(cut["m6_GeV"]),
                    coupling,
                    plan["observation_years"],
                    plan["snr_threshold"],
                )
                partial["samples"].append({**result, "C": coupling})
                if result["decision"] == "unknown":
                    report["unknown_points"].append(
                        {"m6_GeV": cut["m6_GeV"], "C": coupling, **result}
                    )
                checkpoint(store, report)
                return result

            report["cuts"][-1] = trace_cut(cut, evaluate, plan["C_tolerance"])
            checkpoint(store, report)
        report["state"] = (
            "complete_with_unresolved"
            if any(item["state"] != "complete" for item in report["cuts"])
            else "complete"
        )
    print(f"{report['state']}: {store.new_points} pontos novos em {output}", flush=True)
    return 3 if report["state"] == "complete_with_unresolved" else 0
