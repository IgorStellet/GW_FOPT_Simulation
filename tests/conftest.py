"""Ambiente de testes sem janelas e com temporários acessíveis no Windows."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    """Evita o diretório global pytest-of-USER com ACL incompatível.

    Respeita --basetemp e PYTEST_DEBUG_TEMPROOT explícitos. Cada execução
    recebe uma subpasta exclusiva; nunca usamos a pasta de resultados como
    basetemp, pois o pytest apaga o conteúdo desse diretório.
    """
    if (
        os.name == "nt"
        and config.option.basetemp is None
        and not os.environ.get("PYTEST_DEBUG_TEMPROOT")
    ):
        parent = Path(config.rootpath) / "Articles/results/pytest-temp"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = tempfile.mkdtemp(prefix="run-", dir=parent)
