"""Orquestração limitada e retomável; toda a física continua no núcleo v4."""

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

from Articles import boundary_scan as numerical
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


def evaluate_observables_bounded(params, settings, *, numerics, timeout, scratch):
    """Usa o limitador de processos v4 para o novo recorder, sem editar v4."""
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edge-point-", dir=scratch) as temporary:
        folder = Path(temporary)
        request, result, log = [
            folder / name for name in ("request.json", "result.json", "worker.log")
        ]
        request.write_text(
            json.dumps(
                json_safe(
                    {
                        "params": asdict(params),
                        "settings": asdict(settings),
                        "numerics": numerics,
                    }
                ),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        code, timed_out, elapsed = run_with_budget(
            [
                sys.executable,
                "-m",
                "Articles.figures.frontiers.worker",
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
            row = {
                **asdict(params),
                "status": "point_timeout" if timed_out else "numerical_failure",
                "error_type": "PointTimeout" if timed_out else "WorkerExit",
                "error_message": f"Orçamento de {timeout:g}s atingido"
                if timed_out
                else f"Worker saiu com {code}",
            }
            transitions = []
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 32000))
                details = {
                    "worker_log_tail": stream.read().decode("utf-8", errors="replace")
                }
        row.update(
            point_id=point_id(params),
            runtime_seconds=elapsed,
            point_timeout_seconds=timeout,
            completed_utc=datetime.now(UTC).isoformat(),
            calculation_mode="frontier_observables_margin_step",
        )
        return row, transitions, details


class ObservableStore(BoundaryStore):
    """A opção de repetição também abrange beta/GW pendente, com Tn válido.

    O Store original considera apenas a decisão de força ao repetir. Aqui a
    etapa tem outro objetivo: completar observáveis, sem mudar essa decisão.
    A alteração só ocorre no cache da campanha nova e sob --retry-failed.
    """

    def evaluate(self, C, mass):
        if (
            self.stop_requested
            or (self.output / "STOP").exists()
            or (self.output.parent / "STOP").exists()
        ):
            raise BoundaryPaused("Parada solicitada; pontos concluídos publicados.")
        params = ModelParameters(C=C, m6_GeV=float(mass), m8_GeV=self.plan.m8_GeV)
        key = point_id(params)
        old = self.cache.get(key)
        uncomputed = (
            old is not None
            and str(old[0].get("calculation_mode", "")).startswith("boundar")
            and any(item.get("nucleation_accepted") for item in old[1])
            and not any(item.get("beta_over_H") is not None for item in old[1])
        )
        retry = (
            old is not None
            and self.retry_failed
            and key not in self.retried
            and (
                old[0].get("status") == "observables_unresolved"
                or old[0].get("boundary_decision") == "unknown"
            )
        )
        repair = (
            old is None
            or not (self.work / old[0].get("details_path", "missing")).is_file()
        )
        if repair or retry or uncomputed:
            if self.max_points is not None and self.new_points >= self.max_points:
                raise BoundaryPaused(
                    "Orçamento de pontos novos atingido; retome o mesmo plano."
                )
            policy = getattr(self, "numerics", {"policy": "fixed"})
            if policy["policy"] == "metastable_margin_v1":
                result = evaluate_observables_bounded(
                    params,
                    self.settings,
                    numerics=policy,
                    timeout=self.timeout,
                    scratch=self.work / ".point-tmp",
                )
            else:
                result = numerical.evaluate_bounded(
                    params,
                    self.settings,
                    timeout=self.timeout,
                    mode="interior",
                    scratch=self.work / ".point-tmp",
                )
            self.save(*result)
            self.retried.add(key)
            self.new_points += 1
            print(
                f"[{self.new_points}] m8={self.plan.m8_GeV:g}, m6={mass:g}, C={C:g}: {result[0]['status']} ({result[0]['runtime_seconds']:.1f}s)",
                flush=True,
            )
        row = self.cache[key][0]
        return {
            "decision": row["boundary_decision"],
            "reason": row["selection_reason"],
            "status": row["status"],
        }


def execution_manifest(plan, plan_bytes, args, case, stage):
    settings = Settings(**plan["settings"])
    settings.validate()
    numerics = plan.get("numerics", {"policy": "fixed"})
    if numerics.get("policy") not in ("fixed", "metastable_margin_v1"):
        raise ValueError("Política de beta não reconhecida no plano.")
    if numerics["policy"] == "metastable_margin_v1" and (
        not 0 < numerics.get("margin_fraction", 0) <= 0.5
        or not math.isfinite(numerics.get("minimum_step_GeV", 0))
        or numerics.get("minimum_step_GeV", 0) <= 0
    ):
        raise ValueError("Fração/piso de passo inválidos no plano.")
    if not math.isfinite(args.point_timeout) or args.point_timeout <= 0:
        raise ValueError("point-timeout deve ser positivo e finito.")
    if args.max_points is not None and args.max_points < 1:
        raise ValueError("max-points deve ser positivo.")
    modules = ("execution.py", "reach.py", "planning.py", "worker.py")
    grid = {
        "stage": stage,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "task_index": args.task_index,
        "shards_per_scenario": args.shards_per_scenario,
        "m8_GeV": case,
        "point_timeout_seconds": args.point_timeout,
        "beta_numerics": plan.get("numerics", {"policy": "fixed"}),
        "driver_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in modules
        },
    }
    manifest = provenance(settings, grid)
    reuse_compatible(plan["reference_manifest"], manifest, "boundaries")
    return settings, manifest


@contextmanager
def live_store(args, output, settings, manifest, case, report):
    """Um escritor, banco local, sinal de parada e checkpoint por ponto."""
    output, work = output.resolve(), args.work_dir.resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("output e work-dir devem ser separados e não aninhados.")
    validate_campaign(output, manifest)
    if work.exists() and any(work.iterdir()):
        raise ValueError(
            "work-dir precisa estar vazio; a retomada vem do destino persistente."
        )
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    with single_writer(output):
        restore_campaign(output, work)
        store = ObservableStore(
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
            print(
                "Parada solicitada; o ponto atual tem orçamento limitado e será publicado.",
                flush=True,
            )

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


def checkpoint(store, report):
    atomic_json(store.work / "interior_state.json", report)
    store.publish()
