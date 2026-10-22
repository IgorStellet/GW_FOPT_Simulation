"""Checkpoints mantêm dados consistentes e permitem retomada em outro nó."""

import gzip
import json
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from Articles.cluster_runner import copy_campaign, main, restore_campaign
from Articles.collect_data import open_database, save_point


def test_closed_checkpoint_restores_observables_and_ignores_incomplete_files(tmp_path):
    work, persistent, restored = (tmp_path / name for name in ("local", "shared", "new-node"))
    work.mkdir()
    database = open_database(work, {"fingerprint": "fixed-campaign"})
    row = {"point_id": "example", "status": "nucleated"}
    transition = {
        "point_id": "example", "transition_index": 0, "status": "nucleated",
        "alpha_trace": 0.3, "beta_over_H": 120.0, "omega_sw_peak_h2": 1e-11,
    }
    save_point(database, work, (row, [transition], {"observables": transition}))
    database.close()
    (work / "incomplete.json.tmp").write_text("broken")
    copy_campaign(work, persistent)
    assert not (persistent / "incomplete.json.tmp").exists()
    with closing(sqlite3.connect(persistent / "scan.sqlite")) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    restore_campaign(persistent, restored)
    with closing(open_database(restored, {"fingerprint": "fixed-campaign"})) as connection:
        saved = json.loads(connection.execute("SELECT record FROM transitions").fetchone()[0])
        assert saved == transition
    with gzip.open(restored / row["details_path"], "rt", encoding="utf-8") as stream:
        assert json.load(stream)["observables"] == transition


def test_live_wal_is_rejected_without_replacing_previous_checkpoint(tmp_path):
    work, persistent = tmp_path / "local", tmp_path / "shared"
    work.mkdir()
    persistent.mkdir()
    sentinel = persistent / "scan.sqlite"
    sentinel.write_bytes(b"previous confirmed checkpoint")
    database = open_database(work, {"fingerprint": "x"})
    try:
        with pytest.raises(RuntimeError, match="WAL"):
            copy_campaign(work, persistent)
        assert sentinel.read_bytes() == b"previous confirmed checkpoint"
    finally:
        database.close()


def test_runner_rejects_incompatible_checkpoint_without_replacing_saved_data(tmp_path, monkeypatch, capsys):
    import Articles.collect_data as collector

    seed, persistent, work = (tmp_path / name for name in ("seed", "shared", "new-node"))
    seed.mkdir()
    with closing(open_database(seed, {"fingerprint": "previous"})) as database:
        save_point(database, seed, ({"point_id": "kept", "status": "nucleated"}, [], {"kept": True}))
    copy_campaign(seed, persistent)
    saved_database = (persistent / "scan.sqlite").read_bytes()
    monkeypatch.setattr(collector, "provenance", lambda *_: {"fingerprint": "different"})

    def unexpected_point(*args, **kwargs):
        pytest.fail("Uma campanha incompatível não deve calcular pontos.")

    monkeypatch.setattr(collector, "evaluate_point", unexpected_point)
    with pytest.raises(SystemExit) as error:
        main(["--output", str(persistent), "--work-dir", str(work), "--smoke", "--batch-size", "1"])
    assert error.value.code == 2
    assert "Campanha incompatível" in capsys.readouterr().err
    assert (persistent / "scan.sqlite").read_bytes() == saved_database


def test_runner_rejects_overlapping_output_and_work_paths(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path), "--work-dir", str(tmp_path / "scratch")])
    assert error.value.code == 2
    # Esta mensagem é esperada: o teste verifica uma chamada inválida. Capturá-la
    # evita que o editor apresente uma rejeição correta como erro inesperado.
    assert "Pastas distintas e não aninhadas" in capsys.readouterr().err


def test_runner_without_arguments_shows_help_without_starting_scan(tmp_path, monkeypatch, capsys):
    import Articles.cluster_runner as runner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["cluster_runner.py"])

    def unexpected_collect(argv):
        pytest.fail("Ajuda sem argumentos não deve iniciar a coleta.")

    monkeypatch.setattr(runner, "collect_main", unexpected_collect)
    assert runner.main() == 0
    output = capsys.readouterr()
    assert "--smoke" in output.out
    assert "--work-dir" in output.out
    assert output.err == ""
    assert not list(tmp_path.iterdir())


def test_runner_still_rejects_incomplete_arguments(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "shared")])
    assert error.value.code == 2
    assert "--work-dir" in capsys.readouterr().err


def test_runner_reports_unwritable_work_directory_before_calculating(tmp_path, monkeypatch, capsys):
    import Articles.cluster_runner as runner

    original_mkdir = Path.mkdir
    work = tmp_path / "unavailable-scratch"

    def denied(path, *args, **kwargs):
        if path == work:
            raise PermissionError("synthetic unavailable scratch")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(SystemExit) as error:
        runner.main(["--output", str(tmp_path / "shared"), "--work-dir", str(work), "--smoke"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "gravável" in message
    assert "FOPT_SCRATCH_ROOT" in message
    assert "Traceback" not in message
    assert not (tmp_path / "shared").exists()


@pytest.mark.parametrize("selection", ["explicit", "slurm", "tmpdir", "fallback", "invalid"])
def test_slurm_scratch_selection_is_private_and_does_not_create_unavailable_root(tmp_path, selection):
    # Git Bash permite conferir o mesmo helper no Windows; Linux usa o bash
    # instalado. Nenhuma alocação Slurm ou cálculo físico é feita neste teste.
    if os.name == "nt":
        bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if not bash.exists():
            pytest.skip("Git Bash indisponível para testar os templates Slurm.")
    else:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("Bash indisponível para testar os templates Slurm.")
    root = tmp_path / "local-disk"
    root.mkdir()
    missing = tmp_path / "missing-scratch"
    environment = dict(os.environ)
    for key in ("FOPT_SCRATCH_ROOT", "SLURM_TMPDIR", "TMPDIR", "BASH_ENV"):
        environment.pop(key, None)
    if selection == "explicit":
        environment["FOPT_SCRATCH_ROOT"] = root.as_posix()
    elif selection == "slurm":
        environment["SLURM_TMPDIR"] = root.as_posix()
    elif selection == "tmpdir":
        environment["SLURM_TMPDIR"] = missing.as_posix()
        environment["TMPDIR"] = root.as_posix()
    elif selection == "fallback":
        environment["SLURM_TMPDIR"] = missing.as_posix()
        environment["TMPDIR"] = missing.as_posix()
    else:
        environment["FOPT_SCRATCH_ROOT"] = missing.as_posix()
        environment["SLURM_TMPDIR"] = root.as_posix()
    helper = Path(__file__).resolve().parents[1] / "Articles/cluster/job_environment.sh"
    result = subprocess.run(
        [str(bash), "-c", '''
set -euo pipefail
source "$1"
prepare_local_work fopt-test
first="$local_work"
prepare_local_work fopt-test
[[ "$first" != "$local_work" && -d "$first" && -d "$local_work" ]]
if [[ -n "${FOPT_SCRATCH_ROOT:-}${SLURM_TMPDIR:-}${TMPDIR:-}" && "$2" != fallback ]]; then
    [[ "$(cd -- "${first%/*}" && pwd -P)" == "$(cd -- "$3" && pwd -P)" ]]
fi
rmdir -- "$first" "$local_work"
''', "scratch-test", helper.as_posix(), selection, root.as_posix()],
        env=environment, capture_output=True, text=True, encoding="utf-8", timeout=15, check=False,
    )
    assert not missing.exists()
    if selection == "invalid":
        assert result.returncode != 0
        assert "FOPT_SCRATCH_ROOT" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "Diretório temporário do job" in result.stdout
        assert result.stderr == ""


def test_runner_continues_after_repair_that_does_not_increase_point_count(tmp_path, monkeypatch):
    import Articles.cluster_runner as runner

    calls = []

    def fake_collect(argv):
        calls.append(argv)
        folder = tmp_path / "local"
        with closing(open_database(folder, {"fingerprint": "repair-campaign"})) as database:
            if len(calls) <= 2:
                row = {
                    "point_id": "repaired-point", "status": "numerical_failure",
                    "completed_utc": f"2026-09-30T00:00:0{len(calls)}+00:00",
                }
                save_point(database, folder, (row, [], {"attempt": len(calls)}))
        return 0

    monkeypatch.setattr(runner, "collect_main", fake_collect)
    assert runner.main([
        "--output", str(tmp_path / "shared"), "--work-dir", str(tmp_path / "local"),
        "--smoke", "--batch-size", "1",
    ]) == 0
    assert len(calls) == 3  # O segundo lote atualizou uma linha; só o terceiro ficou vazio.
