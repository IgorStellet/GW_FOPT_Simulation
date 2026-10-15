"""Limite real de tempo por ponto, inclusive imports e qualquer recuperação."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from Articles.collect_data import ROOT, json_safe, point_id


def run_with_budget(command, timeout, log_path, cwd=ROOT):
    """wait(timeout) em processo independente; finally não deixa órfãos.

    Arquivo em vez de PIPE: saída grande não bloqueia o processo físico.
    Terminate e kill têm esperas curtas limitadas também no Windows.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("O orçamento por ponto deve ser finito e positivo.")
    started = time.monotonic()
    timed_out = False
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with Path(log_path).open("wb") as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment)
        try:
            process.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
    return process.returncode, timed_out, time.monotonic() - started


def evaluate_bounded(params, settings, *, timeout=300.0, mode="boundaries", scratch=None):
    if mode not in ("boundaries", "interior"):
        raise ValueError("Modo de cálculo inválido.")
    if scratch is not None:
        Path(scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fopt-point-", dir=scratch) as temporary:
        folder = Path(temporary)
        request, result_path, log_path = [folder / name for name in ("request.json", "result.json", "worker.log")]
        request.write_text(json.dumps(json_safe({"params": asdict(params), "settings": asdict(settings),
                                                "mode": mode}), allow_nan=False), encoding="utf-8")
        command = [sys.executable, "-m", "Articles.boundary_physics", "--request", str(request),
                   "--result", str(result_path)]
        code, timed_out, elapsed = run_with_budget(command, timeout, log_path)
        if not timed_out and code == 0 and result_path.exists():
            row, transitions, details = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            row = {**asdict(params), "status": "point_timeout" if timed_out else "numerical_failure",
                   "error_type": "PointTimeout" if timed_out else "WorkerExit",
                   "error_message": f"Orçamento de {timeout:g}s atingido" if timed_out else f"Worker saiu com {code}"}
            transitions = []
            # A leitura é limitada mesmo se uma biblioteca emitir saída excessiva.
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 32000))
                details = {"worker_log_tail": stream.read().decode("utf-8", errors="replace")}
        row.update(point_id=point_id(params), runtime_seconds=elapsed, point_timeout_seconds=timeout,
                   completed_utc=datetime.now(UTC).isoformat(), calculation_mode=mode)
        return row, transitions, details
