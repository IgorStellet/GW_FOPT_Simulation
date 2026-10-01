"""Executa lotes no disco local e publica checkpoints após fechar o SQLite.

Este módulo não implementa física. Chama collect_data.main com a mesma grade,
prescrições e retomada. --output é persistente; --work-dir deve ser local ao
nó. Uma interrupção pode exigir repetir somente o último lote não publicado.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from Articles.collect_data import main as collect_main
from Articles.collect_data import single_writer


def copy_file_atomic(source: Path, destination: Path):
    """Publica um arquivo completo; preserva o anterior durante a cópia."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def copy_campaign(source: Path, destination: Path):
    """Copia campanha fechada, com detalhes primeiro e SQLite por último.

    A cópia persistente usa journal DELETE: não publica WAL/SHM de um banco
    ativo em filesystem compartilhado. Este método exige produtor parado.
    Nenhum arquivo de campanha anterior é apagado.
    """
    database = source / "scan.sqlite"
    if not database.exists():
        return
    wal = source / "scan.sqlite-wal"
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Checkpoint exige SQLite fechado, sem WAL pendente.")
    for path in source.rglob("*"):
        if not path.is_file() or path.name in (
            "scan.sqlite", "scan.sqlite-wal", "scan.sqlite-shm", ".writer.lock"
        ) or path.name.endswith(".tmp"):
            continue
        target = destination / path.relative_to(source)
        if not target.exists() or target.stat().st_mtime_ns != path.stat().st_mtime_ns:
            copy_file_atomic(path, target)
    # Converte somente uma cópia local, preservando o banco de trabalho.
    snapshot = source / "checkpoint.sqlite.tmp"
    shutil.copy2(database, snapshot)
    try:
        with closing(sqlite3.connect(snapshot)) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite não íntegro; checkpoint não publicado.")
        copy_file_atomic(snapshot, destination / "scan.sqlite")
    finally:
        snapshot.unlink(missing_ok=True)


def restore_campaign(source: Path, destination: Path):
    """Restaura um checkpoint persistente para um diretório local novo."""
    if not (source / "scan.sqlite").exists():
        return
    for path in source.rglob("*"):
        if path.is_file() and path.name != ".writer.lock" and not path.name.endswith(".tmp"):
            copy_file_atomic(path, destination / path.relative_to(source))


def point_state(folder: Path) -> tuple[int, str | None]:
    """Conta registros e última atualização, no banco local fechado.

    O horário distingue reparos de detalhes ausentes: esses reparos atualizam
    uma linha existente, sem aumentar a contagem de pontos.
    """
    database = folder / "scan.sqlite"
    if not database.exists():
        return 0, None
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(
            "SELECT count(*), max(json_extract(record, '$.completed_utc')) FROM points"
        ).fetchone()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Teste local (execute na raiz do repositório): "
            "python -m Articles.cluster_runner --smoke --beta-check "
            "--batch-size 1 --workers 2 --output Articles/results/runner_smoke "
            "--work-dir Articles/results/runner_scratch_novo. "
            "No CHE, use Articles/cluster/smoke.slurm."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Pasta persistente da parte.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Pasta NOVA no disco local do nó.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Grade de dois pontos de validação.")
    parser.add_argument("--beta-check", action="store_true")
    arguments = sys.argv[1:] if argv is None else list(argv)
    # O botão Run do editor geralmente não fornece argumentos. Nesse caso,
    # ensina a chamada e termina sem criar pastas ou iniciar a campanha grande.
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    output, work = args.output.resolve(), args.work_dir.resolve()
    if (
        output == work or output in work.parents or work in output.parents
        or args.batch_size < 1 or args.workers < 1
        or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count
    ):
        parser.error("Pastas distintas e não aninhadas; lote/workers/partes positivos e índice válido.")
    if work.exists() and any(work.iterdir()):
        parser.error("work-dir deve estar vazio; restauração virá do checkpoint persistente.")
    try:
        work.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.exit(
            2, f"Não foi possível preparar as pastas: {error}\n"
            "--work-dir deve usar uma pasta local gravável; --output, uma pasta persistente gravável.\n"
            "No Slurm, confira FOPT_SCRATCH_ROOT e FOPT_RESULTS_ROOT.\n",
        )
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Sinal recebido: parar após publicar o lote atual.", flush=True)

    previous_handler = None
    if hasattr(signal, "SIGUSR1"):
        previous_handler = signal.signal(signal.SIGUSR1, request_stop)
    common = [
        "--output", str(work), "--workers", str(args.workers),
        "--max-points", str(args.batch_size),
        "--shard-index", str(args.shard_index), "--shard-count", str(args.shard_count),
    ]
    if args.smoke:
        common += [
            "--m6", "1000", "1000", "5", "--C", "0", "3.35", "3.35",
            "--m8", "668.740304976422", "--no-baselines",
        ]
    if args.beta_check:
        common += ["--beta-check"]
    try:
        with single_writer(output):
            restore_campaign(output, work)
            while not stop_requested:
                before = point_state(work)
                collect_main(common)
                after = point_state(work)
                copy_campaign(work, output)
                print(f"Checkpoint publicado: {after[0]} pontos em {output}", flush=True)
                if after == before:
                    break
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
