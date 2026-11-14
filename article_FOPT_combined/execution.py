"""Processos isolados, limite real por ponto e retomada dos checkpoints."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .core import ROOT
from .model import Potential
from .storage import atomic_json, point_id, point_path, read_json, save_point, task_lock


class Paused(Exception):
    """Orçamento do job/piloto atingido; o próximo comando retoma do checkpoint."""


class Budget:
    """Limita novos pontos e o tempo do coordenador; aceita SIGTERM/USR1 entre pontos."""

    def __init__(self, max_points=None, wall_seconds=None):
        if max_points is not None and max_points < 1:
            raise ValueError("max-points deve ser positivo.")
        if wall_seconds is not None and wall_seconds <= 0:
            raise ValueError("wall-seconds deve ser positivo.")
        self.remaining = max_points
        self.deadline = time.monotonic() + wall_seconds if wall_seconds else None
        self.stopping = False
        self.lock = threading.Lock()

    def reserve(self, timeout):
        """Reserva um ponto apenas se houver tempo para publicá-lo dentro do job."""
        with self.lock:
            if (
                self.stopping
                or self.remaining == 0
                or (self.deadline and time.monotonic() + timeout + 10 >= self.deadline)
            ):
                raise Paused("Pausa com checkpoint: limite de pontos, tempo ou sinal do job.")
            if self.remaining is not None:
                self.remaining -= 1

    def install_signals(self):
        """Solicita parada controlada; o ponto atual termina dentro de seu timeout."""

        def stop(signum, frame):
            self.stopping = True

        for name in ("SIGTERM", "SIGUSR1", "SIGINT"):
            number = getattr(signal, name, None)
            if number is not None:
                signal.signal(number, stop)


def failed_record(parameters, status, message):
    """Falha de execução com decisão desconhecida, distinta de ausência de nucleação."""
    return {
        "parameters": asdict(parameters),
        "transitions": [],
        "context": {},
        "summary": {"status": status, "decision": "unknown", "reason": status},
        "error_type": status,
        "error_message": message,
        "runtime_seconds": 0.0,
    }


def bounded_evaluation(parameters, settings, mode, *, timeout, scratch=None, cached=None):
    """Timeout inclui imports e recuperação; mata o filho e preserva a cauda do log."""
    if timeout <= 0:
        raise ValueError("timeout deve ser positivo.")
    started = time.monotonic()
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    if scratch is not None:
        scratch = Path(scratch)
        if not scratch.is_dir():
            raise ValueError("--scratch deve apontar para uma pasta temporária existente.")
    with tempfile.TemporaryDirectory(prefix="fopt-point-", dir=scratch) as temporary:
        folder = Path(temporary)
        request, target, log = (
            folder / name for name in ("request.json", "result.json", "worker.log")
        )
        atomic_json(
            request,
            {
                "parameters": asdict(parameters),
                "settings": asdict(settings),
                "mode": mode,
                "cached": cached,
            },
        )
        with log.open("wb") as stream:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "article_FOPT_combined.physics",
                    "--request",
                    str(request),
                    "--result",
                    str(target),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            timed_out = False
            try:
                child.wait(timeout=max(0.01, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
        if not timed_out and child.returncode == 0 and target.exists():
            result = read_json(target)
        else:
            result = failed_record(
                parameters,
                "point_timeout" if timed_out else "worker_failure",
                f"Limite de {timeout:g}s" if timed_out else f"Processo saiu com {child.returncode}",
            )
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 32_000))
                result["worker_log_tail"] = stream.read().decode("utf-8", errors="replace")
        result["runtime_seconds"] = time.monotonic() - started
        result["point_timeout_seconds"] = timeout
        return result


class Evaluator:
    """Cache por coordenada e estágio; usado tanto por fronteiras quanto pelo interior."""

    def __init__(
        self, output, settings, mode, budget, *, timeout=300, scratch=None, retry_failed=False
    ):
        self.output, self.settings, self.mode, self.budget = Path(output), settings, mode, budget
        self.timeout, self.scratch, self.retry_failed = timeout, scratch, retry_failed
        self.evaluated = set()

    def __call__(self, parameters):
        """Publica o resultado antes de entregá-lo ao algoritmo que pediu o ponto."""
        path = point_path(self.output, self.mode, parameters)
        identifier = point_id(parameters)
        with task_lock(self.output, f"point-{self.mode}-{identifier}"):
            previous = read_json(path) if path.exists() else None
            # Um ponto completo já contém a nucleação. O primeiro estágio
            # pode aproveitar um piloto anterior sem pagar novamente por Tn.
            complete_path = point_path(self.output, "observables", parameters)
            if previous is None and self.mode == "nucleation" and complete_path.exists():
                complete = read_json(complete_path)
                if complete["summary"]["decision"] != "unknown":
                    previous = deepcopy(complete)
                    previous["reused_observables_result"] = True
                    if previous["summary"]["decision"] == "yes":
                        previous["summary"]["status"] = "nucleated"
                    previous = save_point(self.output, self.mode, parameters, previous)
            failed = previous and (
                previous["summary"]["decision"] == "unknown"
                or (
                    self.mode == "observables"
                    and not previous["summary"].get("observables_complete")
                )
            )
            if previous and (not self.retry_failed or not failed or identifier in self.evaluated):
                return previous
            cached_path = point_path(self.output, "nucleation", parameters)
            cached = (
                read_json(cached_path)
                if self.mode == "observables" and cached_path.exists()
                else None
            )
            if cached and cached["summary"]["decision"] != "yes":
                cached = None
            # A mesma condição necessária do modelo, antes de iniciar Python
            # novamente: milhares de exclusões de vácuo custam milissegundos,
            # sem gastar importações e rastreamento de fases por coordenada.
            started = time.monotonic()
            diagnostics = Potential(parameters).diagnostics()
            if not diagnostics["ew_below_origin"]:
                self.budget.reserve(1.0)
                result = {
                    "parameters": asdict(parameters),
                    "settings": asdict(self.settings),
                    "summary": {
                        "status": "ew_vacuum_unfavoured",
                        "decision": "no",
                        "reason": "V(v,0)>=V(0,0)",
                    },
                    "transitions": [],
                    "context": {},
                    "diagnostics": diagnostics,
                    "fast_vacuum_check": True,
                    "runtime_seconds": time.monotonic() - started,
                }
            else:
                self.budget.reserve(self.timeout)
                result = bounded_evaluation(
                    parameters,
                    self.settings,
                    self.mode,
                    timeout=self.timeout,
                    scratch=self.scratch,
                    cached=cached,
                )
            result = save_point(self.output, self.mode, parameters, result)
            self.evaluated.add(identifier)
            summary = result["summary"]
            print(
                f"{self.mode}: m8={parameters.m8_GeV:g}, m6={parameters.m6_GeV:g}, C={parameters.C:g}: "
                f"{summary['status']} ({result['runtime_seconds']:.1f}s)",
                flush=True,
            )
            return result
