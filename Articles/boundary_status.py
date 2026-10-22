"""Lê progresso, brackets e pendências, sem executar qualquer ponto físico."""

import argparse
import json
from collections import Counter
from pathlib import Path

from Articles.plot_results import load_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    folders = [args.output] if (args.output / "scan.sqlite").exists() else sorted(
        p.parent for p in args.output.glob("m8-*/scan.sqlite"))
    if not folders:
        parser.error("Nenhuma campanha encontrada.")
    results = []
    for folder in folders:
        campaign = load_campaign(folder)
        path = folder / "boundary_state.json"
        if not path.exists():
            path = folder / "interior_state.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"state": "initializing"}
        summary = {"folder": str(folder.resolve()), "state": state["state"], "points": len(campaign["points"]),
                   "statuses": dict(Counter(p["status"] for p in campaign["points"])),
                   "columns": len(state.get("columns", [])),
                   "bounded_columns": sum(c["state"] == "bounded" for c in state.get("columns", [])),
                   "unknown_points": len(state.get("unknown_points", [])), "m6_stop": state.get("m6_stop"),
                   "unresolved_columns": [c["m6_GeV"] for c in state.get("columns", []) if c["state"].startswith("unresolved")]}
        results.append(summary)
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for result in results:
            print(f"{result['folder']}: {result['state']}")
            print(f"  {result['points']} pontos; {result['bounded_columns']}/{result['columns']} colunas delimitadas; "
                  f"{result['unknown_points']} inconclusivos; estados={result['statuses']}")
            if result["unresolved_columns"]:
                print(f"  Colunas pendentes: {result['unresolved_columns'][:10]}")
    return 0 if all(r["state"] in ("complete", "pilot_complete") for r in results) else 3


if __name__ == "__main__":
    raise SystemExit(main())
