"""Diagnóstico somente de leitura: por que a campanha terminou e o que falta."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def summarize(folder):
    folder = Path(folder)
    state = json.loads((folder / "adaptive_state.json").read_text(encoding="utf-8"))
    with (folder / "points.csv").open(encoding="utf-8", newline="") as stream:
        points = list(csv.DictReader(stream))
    unresolved = [{k: p.get(k, "") for k in
                   ("C", "m6_GeV", "m8_GeV", "error_type", "error_message", "details_path")}
                  for p in points if p.get("scan_decision") == "unknown"]
    return {"folder": str(folder.resolve()), "state": state["state"],
            "message": state.get("message"), "points": len(points),
            "decisions": dict(Counter(p.get("scan_decision", "unknown") for p in points)),
            "m6_columns": len({p["m6_GeV"] for p in points}),
            "finite_m6_columns": len({p["m6_GeV"] for p in points if p["m6_GeV"] != "inf"}),
            "C_min_calculated": min((float(p["C"]) for p in points), default=None),
            "C_max_calculated": max((float(p["C"]) for p in points), default=None),
            "unresolved_points": unresolved,
            "mass_stops": [stop for s in state.get("slices", []) for stop in s.get("m6_stops", [])]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    folders = [args.output] if (args.output / "adaptive_state.json").exists() else sorted(
        p.parent for p in args.output.glob("m8-*/adaptive_state.json"))
    if not folders:
        parser.error("Nenhum adaptive_state.json encontrado nesse destino.")
    summaries = [summarize(p) for p in folders]
    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
    else:
        for summary in summaries:
            print(f"{summary['folder']}: {summary['state']}")
            print(f"  {summary['points']} pontos; {summary['m6_columns']} colunas m6 "
                  f"({summary['finite_m6_columns']} finitas); decisões: {summary['decisions']}")
            print(f"  C calculado: {summary['C_min_calculated']}..{summary['C_max_calculated']}; "
                  f"inconclusivos: {len(summary['unresolved_points'])}")
            if summary["message"]:
                print(f"  Motivo: {summary['message']}")
            for stop in summary["mass_stops"]:
                print(f"  Parada em m6: {stop['direction']} / {stop['reason']}")
            for point in summary["unresolved_points"][:5]:
                print(f"  Revisar C={point['C']}, m6={point['m6_GeV']}: "
                      f"{point['error_type']} {point['error_message']} -> {point['details_path']}")
    return 0 if all(s["state"] in ("complete", "pilot_complete") for s in summaries) else 3


if __name__ == "__main__":
    raise SystemExit(main())
