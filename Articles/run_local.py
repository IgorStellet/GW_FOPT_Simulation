"""Três processos locais independentes, um por m8, com pastas separadas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("Articles/results/production_adaptive_v3"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--beta-check", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--C", nargs=3, type=float, default=(0, 10, 0.02))
    parser.add_argument("--m6", nargs=3, type=float, default=(500, 2000, 5))
    parser.add_argument("--seed-C", type=float, default=3.83)
    parser.add_argument("--seed-m6", type=float, default=float("inf"))
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--on-unresolved", choices=("continue", "stop"), default="continue")
    args = parser.parse_args(argv)
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output = args.output.resolve()
    print(f"Modo: {'PILOTO (duas massas)' if args.smoke else 'PRODUÇÃO'}; destino: {args.output}", flush=True)
    common = ["--C", *map(str, args.C), "--m6", *map(str, args.m6),
              "--seed-C", str(args.seed_C), "--seed-m6", str(args.seed_m6),
              "--patience", str(args.patience), "--on-unresolved", args.on_unresolved]
    for name in ("smoke", "plots", "beta_check", "retry_failed"):
        if getattr(args, name):
            common.append("--" + name.replace("_", "-"))
    if args.max_points is not None:
        common += ["--max-points", str(args.max_points)]
    commands = [
        [sys.executable, "-m", "Articles.collect_data", "--scenario-index", str(index),
         "--output", str(args.output / f"m8-{index}"), *common]
        for index in range(3)
    ]
    # Confere as três identidades/configurações antes de iniciar qualquer bounce.
    for command in commands:
        subprocess.run([*command, "--dry-run"], check=True, cwd=ROOT)
    if args.dry_run:
        return 0
    children = []
    try:
        for index, command in enumerate(commands):
            print(f"Iniciando fatia m8-{index}: {args.output / f'm8-{index}'}", flush=True)
            children.append(subprocess.Popen(command, cwd=ROOT))
        codes = [child.wait() for child in children]
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        raise
    unresolved = False
    for index, code in enumerate(codes):
        print(f"Fatia m8-{index}: saída {code}", flush=True)
        state_path = args.output / f"m8-{index}" / "adaptive_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            print(f"  Estado: {state['state']}; inconclusivos: {len(state.get('unresolved_points', []))}; {state_path}", flush=True)
            unresolved |= state["state"].endswith("_with_unresolved") or state["state"] == "unresolved"
    if args.smoke or args.plots:
        from Articles.plot_results import plot_overview

        plot_overview(args.output)
    return 1 if any(codes) else 3 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
