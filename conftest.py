"""Configuração comum a tests/ e Articles/tests/, inclusive no Windows."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    """Escolhe uma pasta exclusiva; respeita temporários explícitos do usuário.

    Na raiz do projeto, este hook também vale quando somente Articles/tests
    é executado. Dentro de tests/ ele não era carregado nessa situação.
    """
    if (
        os.name == "nt"
        and config.option.basetemp is None
        and not os.environ.get("PYTEST_DEBUG_TEMPROOT")
    ):
        parent = Path(config.rootpath) / "Articles/results/pytest-temp"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = tempfile.mkdtemp(prefix="run-", dir=parent)
