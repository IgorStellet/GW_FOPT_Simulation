"""Executa três fatias locais; o scan completo v3 continua em run_local.py."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Teste: python -m Articles.run_boundaries --smoke --output Articles/results/boundaries_pilot_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("boundaries", "interior"), default="boundaries")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if "--scenario-index" in extra or "--work-dir" in extra:
        parser.error("Use Articles.boundary_scan para escolher uma fatia/work-dir; este executor inicia as três.")
    output = (ROOT / (args.output or Path(f"Articles/results/{args.stage}_v1"))).resolve()
    commands = [[sys.executable, "-m", "Articles.boundary_scan", "--stage", args.stage,
                 "--scenario-index", str(index), "--output", str(output / f"m8-{index}"), *extra]
                for index in range(3)]
    # Todas as identidades e fontes de reuso são verificadas antes de iniciar.
    for command in commands:
        subprocess.run([*command, "--dry-run"], cwd=ROOT, check=True)
    if args.dry_run:
        return 0
    # Evita criar a mesma hierarquia simultaneamente em três processos Windows.
    output.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (output / f"m8-{index}").mkdir(parents=True, exist_ok=True)
    print(f"Etapa local: {args.stage}; três fatias independentes; destino={output}", flush=True)
    children = []
    def stop(signum, frame):
        output.mkdir(parents=True, exist_ok=True)
        (output / "STOP").touch()
        print("Parada solicitada nas três fatias. Aguardando o ponto atual limitado por tempo.", flush=True)
    previous = signal.signal(signal.SIGINT, stop)
    try:
        for command in commands:
            children.append(subprocess.Popen(command, cwd=ROOT))
        codes = [child.wait() for child in children]
    except BaseException:
        stop(None, None)
        # Os processos têm um orçamento finito e observam STOP entre pontos.
        for child in children:
            child.wait()
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
    if "--smoke" in extra or "--plots" in extra:
        if args.stage == "boundaries":
            from Articles.plot_boundaries import plot_boundary_overview
            plot_boundary_overview(output)
        else:
            from Articles.plot_results import plot_overview
            plot_overview(output)
    return 1 if any(c not in (0, 3) for c in codes) else 3 if 3 in codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
