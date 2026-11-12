"""Templates novos mantêm uma CPU por fatia e invocam a estratégia nova."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stage", ["boundaries", "interior"])
@pytest.mark.parametrize("smoke", [False, True])
def test_new_templates_keep_resources_and_separate_destinations(stage, smoke):
    path = ROOT / "Articles/cluster" / f"{stage}{'_smoke' if smoke else ''}.slurm"
    text = path.read_text(encoding="utf-8")
    for directive in ("--cpus-per-task=1", "--ntasks=1", "--array=0-2%3", "--partition=milliways-ext",
                      f"--mem={'4' if smoke else '8'}G", f"--time={'02' if smoke else '14'}:00:00"):
        assert directive in text
    assert f"FOPT_STAGE={stage}" in text
    assert f"FOPT_PILOT={int(smoke)}" in text
    assert "boundary_job.sh" in text


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="Sintaxe Bash verificada no nó Linux do CHE")
def test_new_shell_syntax_on_linux():
    folder = ROOT / "Articles/cluster"
    paths = [folder / "boundary_job.sh", *folder.glob("boundaries*.slurm"), *folder.glob("interior*.slurm")]
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=ROOT)
