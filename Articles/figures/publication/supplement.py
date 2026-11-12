"""Plano pequeno para verificar lacunas numéricas sem refazer a campanha.

O plano usa o executor run-edges existente, mas deve ter um destino novo.
Não submete jobs e não altera os bancos de entrada.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from Articles.collect_data import json_safe
from Articles.figures.data import coordinate
from Articles.figures.frontiers.data import read_table
from Articles.figures.frontiers.planning import common_plan, separate_output
from Articles.figures.points import write_plan
from Articles.figures.publication.analysis import refinement_requests
from Articles.figures.publication.inputs import load_original


def build_supplement(dataset, roots, requests, *, beta_step=1.0):
    """Muda somente o passo máximo de beta, mantendo ordem 2 e teste h/2.

    Um passo maior pode reduzir cancelamento/ruído na derivada longe de Tc.
    O limite pela margem metastável e a rejeição por falta de convergência
    continuam ativos. A melhora só é estabelecida após executar o plano.
    """
    if not math.isfinite(beta_step) or beta_step <= 0:
        raise ValueError("beta-step deve ser positivo e finito.")
    by_coordinate = {coordinate(row): row for row in dataset.rows}
    points = {}
    for request in requests:
        key = coordinate(request)
        found = by_coordinate.get(key)
        if found is None or (
            request.get("point_id") and request["point_id"] != found["point_id"]
        ):
            raise ValueError(f"Solicitação não corresponde aos dados originais: {key}")
        points[key] = {
            "m8_GeV": key[0],
            "m6_GeV": key[1],
            "C": key[2],
            "roles": ["publication_numeric_check"],
            "reason": request.get("reason"),
        }
    plan = common_plan(dataset, roots)
    plan["settings"]["beta_step"] = beta_step
    plan.update(
        {
            "schema": "article_point_plan_v1",
            "policy": "publication_targeted_numeric_check",
            "numerics": {
                "policy": "metastable_margin_v1",
                "margin_fraction": 0.1,
                "minimum_step_GeV": 0.005,
            },
            "points": sorted(
                points.values(),
                key=lambda row: (-row["m8_GeV"], -row["m6_GeV"], row["C"]),
            ),
            "edge_targets": [],
            "interpretation": "numeric_check_not_automatic_resolution_of_failed_points",
        }
    )
    return json_safe(plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--observables", type=Path, action="append", default=[])
    parser.add_argument("--reach", type=Path, action="append", default=[])
    parser.add_argument(
        "--requests",
        type=Path,
        help="CSV explícito; sem ele, prioriza lacunas locais LISA.",
    )
    parser.add_argument("--beta-step", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        dataset, profiles = load_original(args.boundaries, args.observables, args.reach)
        requests = (
            read_table(args.requests)
            if args.requests
            else refinement_requests(dataset, profiles)
        )
        plan = build_supplement(
            dataset,
            [args.boundaries, *args.observables, *args.reach],
            requests,
            beta_step=args.beta_step,
        )
        separate_output(args.output, plan)
        write_plan(args.output, plan)
        print(
            f"Plano imutável: {len(plan['points'])} pontos; beta_step máximo={args.beta_step:g} GeV; ordem 2 e teste h/2 preservados."
        )
        print(args.output.resolve())
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
