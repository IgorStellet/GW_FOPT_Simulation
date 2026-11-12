"""Planeja e calcula somente coordenadas escolhidas, usando o núcleo v4.

O plano é imutável. Os resultados têm outro destino; nenhuma campanha de
fronteiras é escrita. BoundaryStore fornece o SQLite local, retomada e cópia
fechada já usados no cluster, e evaluate_bounded conserva o teto por ponto.
"""

from __future__ import annotations

import hashlib
import json
import math
import signal
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from Articles.boundary_scan import (
    BoundaryStore,
    atomic_json,
    restore_campaign,
    reuse_compatible,
)
from Articles.boundary_search import BoundaryPaused, BoundaryPlan
from Articles.collect_data import (
    M8_SCENARIOS_GEV,
    Settings,
    json_safe,
    provenance,
    single_writer,
    validate_campaign,
)
from Articles.figures.data import coordinate, load_dataset


def build_plan(dataset, *, masses=3, samples=3, controls=False):
    """Amostragem estratificada de pontos JÁ selecionados; inclui m6=infinito.

    Um mapa denso demanda valores físicos adicionais. Este plano não inventa
    alpha/beta nos pontos omitidos nem promete resolver um mapa com 9 pontos.
    """
    if masses < 1 or samples < 1:
        raise ValueError("mass-samples e C-samples devem ser positivos.")
    targets = {}
    for case in dataset.cases:
        positive = [row for row in dataset.rows if float(row["m8_GeV"]) == case and row["boundary_decision"] == "yes"]
        available = sorted({float(row["m6_GeV"]) for row in positive})
        chosen = [available[i] for i in sorted(set(np.linspace(0, len(available)-1, min(masses, len(available)), dtype=int)))] if available else []
        for mass in chosen:
            rows = sorted([row for row in positive if float(row["m6_GeV"]) == mass], key=lambda row: row["C"])
            for index in sorted(set(np.linspace(0, len(rows)-1, min(samples, len(rows)), dtype=int))):
                row = rows[index]
                if not row.get("gw_resolved"):
                    targets[coordinate(row)] = {"m6_GeV": mass, "m8_GeV": case, "C": row["C"], "purpose": "GW_and_thermodynamics"}
                if controls:
                    for key, purpose in (((case, mass, 0.0), "polynomial_control"), ((math.inf, math.inf, float(row["C"])), "measure_control")):
                        existing = next((item for item in dataset.rows if coordinate(item) == key), None)
                        if existing is None or existing["boundary_decision"] == "unknown" or existing["boundary_decision"] == "yes" and not existing.get("gw_resolved"):
                            targets[key] = {"m6_GeV": key[1], "m8_GeV": key[0], "C": key[2], "purpose": purpose}
    reference = dataset.campaigns[0]["manifest"]
    settings = {**asdict(Settings()), **reference.get("settings", {}), "beta_check": True, "recovery_attempts": 0}
    return json_safe({"schema": "article_point_plan_v1", "created_utc": datetime.now(UTC).isoformat(),
                      "reference_manifest": reference, "settings": settings,
                      "points": sorted(targets.values(), key=lambda row: (-float(row["m8_GeV"]), -float(row["m6_GeV"]), row["C"])),
                      "source_fingerprints": [campaign["manifest"].get("fingerprint") for campaign in dataset.campaigns],
                      "policy": "stratified_existing_selected_points_plus_exact_pure_controls"})


def write_plan(path, plan):
    path = Path(path)
    if path.exists():
        raise ValueError(f"O plano já existe e foi preservado: {path}. Escolha outro nome para um plano novo.")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, plan)


def run_points(args):
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema") != "article_point_plan_v1":
        raise ValueError("Formato de plano não reconhecido.")
    points = plan["points"]
    if args.scenario_index is not None:
        case = M8_SCENARIOS_GEV[args.scenario_index]
        points = [point for point in points if float(point["m8_GeV"]) == case]
    cases = {float(point["m8_GeV"]) for point in points}
    if len(cases) > 1:
        raise ValueError("Escolha --scenario-index 0/1/2; cada job escreve uma única fatia.")
    if not points:
        print("Esta fatia não tem pontos pendentes no plano.", flush=True)
        return 0
    if not math.isfinite(args.point_timeout) or args.point_timeout <= 0:
        raise ValueError("point-timeout deve ser positivo e finito.")
    if args.max_points is not None and args.max_points < 1:
        raise ValueError("max-points deve ser positivo.")
    settings = Settings(**plan["settings"])
    settings.validate()
    grid = {"stage": "article_selected_points", "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "point_timeout_seconds": args.point_timeout, "m8_GeV": next(iter(cases)),
            "points": points}
    manifest = provenance(settings, grid)
    reuse_compatible(plan["reference_manifest"], manifest, "boundaries")
    validate_campaign(args.output, manifest)
    if args.dry_run:
        print(f"Plano compatível: {len(points)} pontos nesta fatia; nenhum cálculo/pasta nova.")
        return 0
    output, work = args.output.resolve(), args.work_dir.resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("output e work-dir devem ser distintos e não aninhados.")
    if work.exists() and any(work.iterdir()):
        raise ValueError("work-dir deve estar vazio; a retomada vem do destino persistente.")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    boundary_plan = BoundaryPlan(m8_GeV=next(iter(cases)))
    report = {"state": "running", "points_planned": len(points), "points_checked": 0,
              "plan_sha256": grid["plan_sha256"], "unknown_points": []}
    with single_writer(output):
        restore_campaign(output, work)
        store = BoundaryStore(work, output, manifest, settings, boundary_plan, "interior", args.point_timeout,
                              args.max_points, args.retry_failed)
        previous = {}

        def stop(signum, frame):
            store.stop_requested = True
            print("Parada solicitada: publicar após o ponto atual limitado.", flush=True)

        for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous[number] = signal.signal(number, stop)
        try:
            try:
                for point in points:
                    result = store.evaluate(float(point["C"]), float(point["m6_GeV"]))
                    report["points_checked"] += 1
                    if result["decision"] == "unknown" or result["status"] == "observables_unresolved":
                        report["unknown_points"].append(point)
                    atomic_json(work / "interior_state.json", report)
                    store.publish()
                report["state"] = "complete_with_unresolved" if report["unknown_points"] else "complete"
            except BoundaryPaused as error:
                report.update(state="paused", message=str(error))
            except BaseException:
                report["state"] = "interrupted"
                raise
            finally:
                atomic_json(work / "interior_state.json", report)
                store.export()
                store.publish()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    print(f"{report['state']}: {store.new_points} pontos calculados; dados em {output}", flush=True)
    return 3 if report["state"] == "complete_with_unresolved" else 0


def plan_from_args(args):
    dataset = load_dataset(args.boundaries, args.observables, read_details=False,
                           allow_unverified=args.allow_unverified)
    plan = build_plan(dataset, masses=args.mass_samples, samples=args.C_samples, controls=args.include_controls)
    if args.point:
        from Articles.combined_model import ModelParameters

        explicit = {}
        for mass, coupling, case in args.point:
            parameters = ModelParameters(m6_GeV=mass, C=coupling, m8_GeV=case)
            if case not in M8_SCENARIOS_GEV:
                raise ValueError("O plano do cluster usa os três cenários m8 da campanha: inf, 840.8964152537145 e 668.740304976422.")
            explicit[coordinate(asdict(parameters))] = {"m6_GeV": mass, "m8_GeV": case, "C": coupling, "purpose": "explicit_request"}
        plan["points"] = json_safe(list(explicit.values()))
        plan["policy"] = "explicit_coordinates_only"
    write_plan(args.output, plan)
    print(f"Plano salvo: {len(plan['points'])} coordenadas em {args.output.resolve()}")
    return 0
