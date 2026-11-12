"""Execução limitada do plano de cobertura com os checkpoints existentes."""

from __future__ import annotations

import hashlib
import json
import math
import signal
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from Articles.boundary_scan import (
    BoundaryStore,
    atomic_json,
    restore_campaign,
    reuse_compatible,
)
from Articles.boundary_search import BoundaryPaused, BoundaryPlan
from Articles.bounded_evaluation import run_with_budget
from Articles.collect_data import (
    Settings,
    json_safe,
    point_id,
    provenance,
    single_writer,
    validate_campaign,
)
from Articles.combined_model import ModelParameters
from Articles.figures.data import campaign_folders, load_dataset
from Articles.figures.frontiers.execution import checkpoint
from Articles.figures.frontiers.planning import separate_output
from Articles.figures.publication.coverage import (
    complete_observables,
    complete_transition_set,
)


def evaluate_bounded(params, settings, *, numerics, timeout, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="coverage-point-", dir=scratch
    ) as temporary:
        root = Path(temporary)
        request, result, log = (
            root / name for name in ("request.json", "result.json", "worker.log")
        )
        request.write_text(
            json.dumps(
                json_safe(
                    {
                        "params": asdict(params),
                        "settings": asdict(settings),
                        "numerics": numerics,
                    }
                )
            ),
            encoding="utf-8",
        )
        code, timed_out, elapsed = run_with_budget(
            [
                sys.executable,
                "-m",
                "Articles.figures.publication.coverage_worker",
                "--request",
                str(request),
                "--result",
                str(result),
            ],
            timeout,
            log,
        )
        if not timed_out and code == 0 and result.is_file():
            row, transitions, details = json.loads(result.read_text(encoding="utf-8"))
        else:
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 32000))
                tail = stream.read().decode("utf-8", errors="replace")
            row = {
                **asdict(params),
                "status": "point_timeout" if timed_out else "numerical_failure",
                "error_type": "PointTimeout" if timed_out else "WorkerExit",
                "error_message": f"Orçamento de {timeout:g}s atingido"
                if timed_out
                else f"Worker saiu com {code}",
            }
            transitions, details = [], {"worker_log_tail": tail}
        row.update(
            point_id=point_id(params),
            runtime_seconds=elapsed,
            point_timeout_seconds=timeout,
            completed_utc=datetime.now(UTC).isoformat(),
            calculation_mode="coverage_observables_v2",
        )
        return row, transitions, details


class CoverageStore(BoundaryStore):
    def evaluate(self, C, mass):
        if (
            self.stop_requested
            or (self.output / "STOP").exists()
            or (self.output.parent / "STOP").exists()
        ):
            raise BoundaryPaused("Parada solicitada; checkpoints preservados.")
        params = ModelParameters(C=C, m6_GeV=float(mass), m8_GeV=self.plan.m8_GeV)
        key = point_id(params)
        old = self.cache.get(key)
        missing_details = (
            old is not None
            and not (self.work / old[0].get("details_path", "__missing__")).is_file()
        )
        retry = (
            old is not None
            and self.retry_failed
            and key not in self.retried
            and not complete_transition_set(old[1])
            and old[0].get("status")
            not in ("no_first_order_found", "no_nucleation_found")
        )
        if old is None or retry or missing_details:
            if self.max_points is not None and self.new_points >= self.max_points:
                raise BoundaryPaused(
                    "Orçamento de pontos atingido; retome o mesmo plano."
                )
            result = evaluate_bounded(
                params,
                self.settings,
                numerics=self.numerics,
                timeout=self.timeout,
                scratch=self.work / ".point-tmp",
            )
            self.save(*result)
            self.new_points += 1
            self.retried.add(key)
            print(
                f"[{self.new_points}] m8={params.m8_GeV:g}, m6={mass:g}, C={C:g}: {result[0]['status']} ({result[0]['runtime_seconds']:.1f}s)",
                flush=True,
            )
        return self.cache[key]


@contextmanager
def coverage_store(args, output, manifest, settings, case, report):
    output, work = output.resolve(), args.work_dir.resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("output e work-dir devem ser separados e não aninhados.")
    if work.exists() and any(work.iterdir()):
        raise ValueError(
            "Use work-dir vazio; a retomada vem do checkpoint persistente."
        )
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    with single_writer(output):
        restore_campaign(output, work)
        store = CoverageStore(
            work,
            output,
            manifest,
            settings,
            BoundaryPlan(m8_GeV=case),
            "interior",
            args.point_timeout,
            args.max_points,
            args.retry_failed,
        )
        store.numerics = manifest["grid"]["beta_numerics"]
        previous = {}

        def stop(signum, frame):
            store.stop_requested = True

        for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous[number] = signal.signal(number, stop)
        try:
            try:
                yield store
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


def run_shard(args):
    content = args.plan.read_bytes()
    plan = json.loads(content)
    if plan.get("schema") != "article_coverage_plan_v2":
        raise ValueError("Use um plano article_coverage_plan_v2.")
    cases = [float(case) for case in plan["cases_GeV"]]
    shards = args.shards_per_scenario
    if shards < 1 or not 0 <= args.task_index < len(cases) * shards:
        raise ValueError("Índice ou número de partes inválido.")
    if (
        not math.isfinite(args.point_timeout)
        or args.point_timeout <= 0
        or args.max_points is not None
        and args.max_points < 1
    ):
        raise ValueError("Orçamentos devem ser positivos.")
    scenario, part = divmod(args.task_index, shards)
    case = cases[scenario]
    points = [p for p in plan["points"] if float(p["m8_GeV"]) == case][part::shards]
    separate_output(args.output, plan)
    work, output_root = args.work_dir.resolve(), args.output.resolve()
    if (
        work == output_root
        or work in output_root.parents
        or output_root in work.parents
    ):
        raise ValueError("output e work-dir devem ser separados e não aninhados.")
    settings = Settings(**plan["settings"])
    settings.validate()
    numerics = plan["numerics"]
    if (
        numerics.get("policy") != "central_step_ladder_v1"
        or not numerics.get("maximum_steps_GeV")
        or any(not math.isfinite(h) or h <= 0 for h in numerics["maximum_steps_GeV"])
    ):
        raise ValueError("Política/passos de beta inválidos.")
    if (
        not math.isfinite(numerics["margin_fraction"])
        or not 0 < numerics["margin_fraction"] <= 0.5
        or not math.isfinite(numerics["minimum_step_GeV"])
        or numerics["minimum_step_GeV"] <= 0
    ):
        raise ValueError("Margem/piso de beta inválidos.")
    files = ("coverage_execution.py", "coverage_worker.py")
    grid = {
        "stage": "article_coverage_observables_v2",
        "plan_sha256": hashlib.sha256(content).hexdigest(),
        "task_index": args.task_index,
        "shards_per_scenario": shards,
        "m8_GeV": case,
        "point_timeout_seconds": args.point_timeout,
        "beta_numerics": numerics,
        "driver_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in files
        },
        "margin_recorder_sha256": hashlib.sha256(
            Path(__file__).parents[1].joinpath("frontiers", "worker.py").read_bytes()
        ).hexdigest(),
    }
    manifest = provenance(settings, grid)
    reuse_compatible(plan["reference_manifest"], manifest, "boundaries")
    output = args.output / f"m8-{scenario}-part-{part}"
    validate_campaign(output, manifest)
    if args.dry_run or not points:
        print(
            f"Tarefa {args.task_index}: {len(points)} pontos; configuração compatível; nenhum cálculo/pasta nova."
        )
        return 0
    report = {
        "kind": "coverage_observables",
        "state": "running",
        "points_planned": len(points),
        "points_checked": 0,
        "points_with_complete_primary_observables": 0,
        "points_with_resolved_negative_search": 0,
        "unknown_points": [],
        "task_index": args.task_index,
        "plan_sha256": grid["plan_sha256"],
        "m8_GeV": json_safe(case),
    }
    with coverage_store(args, output, manifest, settings, case, report) as store:
        # Importa somente observáveis completos dos pilotos, usando o leitor e
        # o copiador de detalhes já auditados; fontes originais ficam intactas.
        allowed = {(float(p["C"]), float(p["m6_GeV"])) for p in points}
        for root in args.reuse:
            source = load_dataset(root, read_details=True)
            reusable = {
                (r["C"], float(r["m6_GeV"]))
                for r in source.rows
                if float(r["m8_GeV"]) == case and complete_observables(r)
            }
            store.allowed_coordinates = allowed & reusable
            for folder in campaign_folders(root):
                store.import_campaign(folder)
        store.allowed_coordinates = allowed
        for point in points:
            row, transitions = store.evaluate(float(point["C"]), float(point["m6_GeV"]))
            report["points_checked"] += 1
            complete = complete_transition_set(transitions)
            negative = (
                row.get("status") in ("no_first_order_found", "no_nucleation_found")
                and not transitions
            )
            if not complete and not negative:
                report["unknown_points"].append({**point, "status": row.get("status")})
            report["points_with_complete_primary_observables"] += int(complete)
            report["points_with_resolved_negative_search"] += int(negative)
            checkpoint(store, report)
        report["state"] = (
            "complete_with_unresolved" if report["unknown_points"] else "complete"
        )
    print(
        f"{report['state']}: {report['points_checked']}/{len(points)} pontos em {output}",
        flush=True,
    )
    return 0  # Falhas de pontos são dados; erros de execução continuam fatais.
