"""Temporários exclusivos dentro do pacote, evitando o Temp bloqueado no Windows."""

import tempfile
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        root = Path(__file__).resolve().parents[1] / ".work" / "pytest"
        root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = tempfile.mkdtemp(prefix="run-", dir=root)
