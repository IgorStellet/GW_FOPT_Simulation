"""Checkpoint por ponto, manifesto físico e exportação; nenhuma base antiga é importada.

Cada resultado é um JSON publicado por rename atômico na pasta persistente.
As tentativas anteriores permanecem em attempts/. CSVs são produtos derivados,
recriáveis a qualquer momento. Um lock por tarefa evita dois escritores iguais.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import campaign_from_dict
from .core import PACKAGE, ROOT
from .model import MODEL_VERSION, Parameters

SCHEMA = 1


def safe(value):
    """JSON estrito: inf identifica operador desligado; NaN/-inf tornam-se null."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value == math.inf else None
    return value


def canonical(value):
    """Serialização determinística usada nos hashes e nos arquivos salvos."""
    return json.dumps(safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def timestamp():
    """Horário UTC explícito para checkpoints, tentativas e auditoria."""
    return datetime.now(UTC).isoformat()


def point_id(parameters):
    """Identificador das coordenadas; a identidade física está no manifesto da pasta."""
    return hashlib.sha256(canonical(asdict(parameters)).encode()).hexdigest()[:24]


def parameters_from_dict(data):
    """Converte apenas as três coordenadas físicas de um registro salvo."""
    return Parameters(**{k: math.inf if v == "inf" else float(v) for k, v in data.items()})


def atomic_json(path, value):
    """Escreve, sincroniza e publica no mesmo filesystem, preservando o arquivo anterior em falhas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    """Lê um produto JSON; erros de integridade permanecem visíveis ao usuário."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def physics_identity():
    """Hash apenas do código físico, seus dados spline e versões numéricas.

    Editar legendas, Slurm ou figuras não invalida um scan. Alterar potencial,
    núcleo, parâmetros numéricos ou versões exige uma nova pasta de campanha.
    """
    files = [PACKAGE / name for name in ("model.py", "core.py", "physics.py", "config.py")]
    core = ROOT / "src" / "CosmoTransitions"
    files += sorted(core.glob("*.py")) + sorted(core.glob("*.npz"))
    return {
        "model": MODEL_VERSION,
        "files": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
    }


def initialize(output, campaign):
    """Cria uma campanha nova ou verifica compatibilidade integral de uma existente."""
    campaign.validate()
    output = Path(output).resolve()
    identity = {"schema": SCHEMA, "campaign": asdict(campaign), "physics": physics_identity()}
    fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
    path = output / "manifest.json"
    if path.exists():
        saved = read_json(path)
        if saved["fingerprint"] != fingerprint:
            raise ValueError(
                "Campanha incompatível: configuração, código físico ou versões mudaram. "
                "Os dados foram preservados; escolha outra pasta --output."
            )
        return saved
    if output.exists() and any(output.iterdir()):
        raise ValueError("Pasta não vazia sem manifesto desta campanha; escolha uma pasta nova.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **identity,
        "fingerprint": fingerprint,
        "created_utc": timestamp(),
        "units": {"field": "GeV", "temperature": "GeV", "potential": "GeV^4", "frequency": "Hz"},
        "approximation": "one-loop + AE rings; scalar hard self-energies T2; no T4/T6 flowers",
        "alpha_for_GW": "signed_trace_anomaly",
        "GW_source": "sound_waves",
        "selection": "accepted first-order nucleation, |phi_low(Tn)|/Tn > threshold; V(v,0)<V(0,0)",
        "boundary_assumption": "one connected selected interval in C; additional sampled islands flag the column",
        "workflow_source_sha256": {
            name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
            for name in ("boundaries.py", "observables.py", "execution.py", "storage.py")
        },
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    atomic_json(path, manifest)
    return manifest


def load_campaign(output, *, verify=True):
    """Abre uma campanha; os leitores gráficos podem dispensar comparação com o código atual."""
    manifest = read_json(Path(output) / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Schema de campanha desconhecido.")
    campaign = campaign_from_dict(manifest["campaign"])
    if verify:
        initialize(output, campaign)
    return campaign, manifest


@contextmanager
def task_lock(output, name):
    """Lock de sistema operacional; é liberado em crash, sem apagar checkpoints."""
    path = Path(output) / "locks" / f"{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Outra tarefa está usando o lock {name}.") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def point_path(output, mode, parameters):
    """Caminho de um resultado; estágios não sobrescrevem os resultados uns dos outros."""
    return Path(output) / "points" / mode / f"{point_id(parameters)}.json"


def save_point(output, mode, parameters, result):
    """Preserva cada tentativa e publica o último resultado do estágio."""
    result = {
        **result,
        "point_id": point_id(parameters),
        "completed_utc": timestamp(),
        "mode": mode,
    }
    attempts = Path(output) / "attempts" / mode / point_id(parameters)
    attempts.mkdir(parents=True, exist_ok=True)
    atomic_json(attempts / f"{len(list(attempts.glob('*.json'))):04d}.json", result)
    atomic_json(point_path(output, mode, parameters), result)
    return safe(result)


def point_records(output, mode):
    """Itera os resultados físicos mais recentes, sem executar cálculos."""
    return [read_json(path) for path in sorted((Path(output) / "points" / mode).glob("*.json"))]


def write_csv(path, rows):
    """Publica CSV UTF-8 com união de colunas; estruturas ficam como JSON em células."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: canonical(v) if isinstance(v, (dict, list, tuple)) else safe(v)
                        for k, v in row.items()
                    }
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_tables(output):
    """Exporta pontos, todas as transições e fronteiras; dispensa refazer a física."""
    destination = Path(output) / "tables"
    points, transitions_rows = [], []
    for mode in ("nucleation", "observables"):
        for record in point_records(output, mode):
            common = {"point_id": record["point_id"], "mode": mode, **record["parameters"]}
            points.append(
                {
                    **common,
                    **record["summary"],
                    "runtime_seconds": record.get("runtime_seconds"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                }
            )
            transitions_rows.extend({**common, **row} for row in record.get("transitions", []))
    boundaries = [
        read_json(p) for p in sorted((Path(output) / "boundaries").glob("m8-*/column-*.json"))
    ]
    write_csv(destination / "points.csv", points)
    write_csv(destination / "transitions.csv", transitions_rows)
    write_csv(destination / "boundaries.csv", boundaries)
    return {"points": len(points), "transitions": len(transitions_rows), "columns": len(boundaries)}


def archive_campaign(output, destination):
    """ZIP atômico dos dados publicados, sem locks de execução nem arquivos temporários.

    Excluir locks é necessário também no Windows, onde um byte bloqueado por
    um coordenador ativo impede a leitura do arquivo por outro processo.
    Para um snapshot global coerente, exporte depois de todos os jobs terminarem.
    """
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if destination.suffix.lower() != ".zip" or output in destination.parents:
        raise ValueError("O ZIP deve terminar em .zip e ficar fora da campanha.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".zip", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                relative = path.relative_to(output)
                if (
                    not path.is_file()
                    or "locks" in relative.parts
                    or path.suffix not in (".json", ".csv")
                    or any(part.startswith(".") for part in relative.parts)
                    or output not in path.resolve().parents
                ):
                    continue
                archive.write(path, str(relative).replace("\\", "/"))
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
