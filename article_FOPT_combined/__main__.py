"""Interface única: init -> boundaries -> observables -> figures -> export."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from .config import Campaign
from .execution import Budget, Evaluator, Paused
from .model import Parameters
from .storage import (
    archive_campaign,
    atomic_json,
    export_tables,
    initialize,
    load_campaign,
    point_id,
    point_records,
    read_json,
    safe,
    task_lock,
)


def parser():
    """Declara somente opções usadas pelo fluxo atual; --help explica cada comando."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Cria uma configuração imutável e um manifesto.")
    init.add_argument(
        "--output", type=Path, required=True, help="Pasta nova e persistente de campanha."
    )
    init.add_argument("--preset", choices=("pilot", "full"), default="pilot")
    init.add_argument("--m6", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"))
    init.add_argument("--m8", type=float, nargs="+", help="Massas em GeV; inf desliga o operador.")
    init.add_argument("--C", type=float, nargs=2, metavar=("MIN", "MAX"))
    init.add_argument("--C-tolerance", type=float)
    init.add_argument("--interior-points", type=int)
    init.add_argument("--interior-stride", type=int)
    init.add_argument(
        "--beta-step",
        type=float,
        help="Passo máximo de beta em GeV; reduções próximas aos extremos são validadas.",
    )
    init.add_argument(
        "--fixed-beta-step",
        action="store_true",
        help="Desativa a redução validada do passo perto dos limites de fase.",
    )
    init.add_argument(
        "--action-tolerance", type=float, help="Residual máximo de S3/T em torno de 140."
    )
    init.add_argument("--T-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    init.add_argument("--n-phi", type=int)
    init.add_argument("--dry-run", action="store_true", help="Mostra configuração sem criar pasta.")
    for name, help_text in (
        ("boundaries", "Busca apenas nucleação e bordas."),
        ("observables", "Calcula alpha, beta, Tc e GW entre as bordas."),
        ("point", "Calcula coordenadas específicas, sem refazer campanha."),
    ):
        stage = sub.add_parser(name, help=help_text)
        stage.add_argument("--output", type=Path, required=True)
        stage.add_argument(
            "--timeout",
            type=float,
            default=300 if name == "boundaries" else 900,
            help="Orçamento real em segundos por ponto, incluindo recuperação.",
        )
        stage.add_argument("--max-points", type=int, help="Limite de pontos novos neste comando.")
        stage.add_argument(
            "--wall-seconds", type=float, help="Tempo do coordenador, menor que o limite Slurm."
        )
        stage.add_argument(
            "--scratch",
            type=Path,
            help="Pasta temporária existente; padrão é o temporário do sistema.",
        )
        stage.add_argument(
            "--retry-failed",
            action="store_true",
            help="Retenta falhas/observáveis pendentes uma vez nesta execução.",
        )
        stage.add_argument(
            "--dry-run",
            action="store_true",
            help="Confere configuração e mostra plano sem calcular.",
        )
        if name != "point":
            stage.add_argument("--shard-index", type=int, default=0)
            stage.add_argument("--shard-count", type=int, default=1)
        if name == "observables":
            stage.add_argument("--workers", type=int, default=1)
            stage.add_argument(
                "--allow-partial",
                action="store_true",
                help="Usa bordas já disponíveis; reservado ao piloto.",
            )
            stage.add_argument(
                "--refine",
                action="store_true",
                help="Dobra a resolução em C, aproveitando o cache.",
            )
        if name == "point":
            stage.add_argument("--m6", type=float, default=math.inf)
            stage.add_argument("--m8", type=float, default=math.inf)
            stage.add_argument("--C", type=float)
            stage.add_argument("--points-file", type=Path, help="CSV com m6_GeV,m8_GeV,C.")
    status = sub.add_parser("status", help="Lê progresso e pendências, sem calcular.")
    status.add_argument("--output", type=Path, required=True)
    check = sub.add_parser(
        "check-pilot", help="Exige ao menos um ponto com observáveis completos em cada m8."
    )
    check.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export", help="Atualiza CSVs e cria um ZIP portátil.")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--zip", type=Path, help="Nome do ZIP fora da pasta de campanha.")
    figures = sub.add_parser(
        "figures", help="Desenha as seis figuras apenas a partir dos resultados salvos."
    )
    figures.add_argument("--output", type=Path, required=True, help="Pasta da campanha.")
    figures.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Pasta separada para as figuras e tabelas gráficas.",
    )
    figures.add_argument(
        "--m8",
        type=float,
        nargs="+",
        help="Três cenários principais; o quarto ganha atlas suplementar.",
    )
    figures.add_argument("--years", type=float, default=4)
    figures.add_argument("--snr", type=float, default=10)
    figures.add_argument("--dpi", type=int, default=300)
    figures.add_argument(
        "--draft", action="store_true", help="Permite produtos parciais, marcados como DRAFT."
    )
    figures.add_argument(
        "--replace", action="store_true", help="Regenera somente produtos gráficos desta edição."
    )
    return p


def init_campaign(args):
    """Traduz o preset e as poucas opções públicas em uma configuração verificável."""
    if args.preset == "pilot":
        campaign = Campaign(
            m6_range=(1200.0, 1200.0, 5.0),
            C_tolerance=0.05,
            seed_resolution=0.1,
            interior_points=3,
            interior_stride=1,
        )
    else:
        campaign = Campaign()
    values = {}
    for argument, name in (("m6", "m6_range"), ("m8", "m8_values"), ("C", "C_range")):
        value = getattr(args, argument)
        if value is not None:
            values[name] = tuple(value)
    for name in ("C_tolerance", "interior_points", "interior_stride"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    numerical = {}
    for argument, name in (
        ("beta_step", "beta_step_GeV"),
        ("action_tolerance", "action_tolerance"),
        ("n_phi", "n_phi"),
    ):
        value = getattr(args, argument)
        if value is not None:
            numerical[name] = value
    if args.T_range:
        numerical.update(T_min=args.T_range[0], T_max=args.T_range[1])
    if args.fixed_beta_step:
        numerical["adaptive_beta"] = False
    campaign = replace(campaign, **values, numerics=replace(campaign.numerics, **numerical))
    campaign.validate()
    print(safe(asdict(campaign)))
    print(
        f"{len(campaign.masses()) * len(campaign.m8_values)} colunas de fronteira; "
        f"até {len(campaign.observable_masses()) * len(campaign.m8_values) * (campaign.interior_points + 2)} "
        "pontos-base de observáveis, mais validação."
    )
    if not args.dry_run:
        initialize(args.output, campaign)
        print(f"Campanha preparada: {args.output.resolve()}")


def status(output):
    """Resumo por m8; complete_with_unresolved nunca é anunciado como validação científica."""
    campaign, manifest = load_campaign(output, verify=False)
    report = {"fingerprint": manifest["fingerprint"], "cases": []}
    observables = point_records(output, "observables")
    nucleation = point_records(output, "nucleation")
    from .observables import plan

    planned = plan(
        output,
        campaign,
        allow_partial=True,
        refine=(Path(output) / "plans" / "observables-refined.json").exists(),
    )
    for ci, case in enumerate(campaign.m8_values):
        columns = [
            read_json(p)
            for p in sorted((Path(output) / "boundaries" / f"m8-{ci}").glob("column-*.json"))
        ]
        obs = [r for r in observables if r["parameters"]["m8_GeV"] == safe(case)]
        targets = [p for p in planned["points"] if p["m8_GeV"] == case]
        saved_ids = {r["point_id"] for r in obs}
        entry = {
            "m8_GeV": case,
            "columns_done": len(columns),
            "columns_expected": len(campaign.masses()),
            "states": dict(Counter(r["state"] for r in columns)),
            "observables_saved": len(obs),
            "observables_complete": sum(
                r["summary"].get("observables_complete", False) for r in obs
            ),
            "nucleation_points_saved": sum(
                r["parameters"]["m8_GeV"] == safe(case) for r in nucleation
            ),
            "observables_planned": len(targets),
            "observables_missing": sum(p["point_id"] not in saved_ids for p in targets),
        }
        report["cases"].append(entry)
        print(safe(entry))
    atomic_json(Path(output) / "status.json", report)
    return report


def execute(args):
    """Abre uma campanha compatível, executa o estágio e publica tabelas ao terminar."""
    from . import boundaries, observables

    campaign, manifest = load_campaign(args.output)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout deve ser positivo e finito.")
    if args.command == "boundaries":
        requests = list(boundaries.assigned_columns(campaign, args.shard_index, args.shard_count))
        print(
            f"Fronteiras: {len(requests)} colunas atribuídas; fingerprint={manifest['fingerprint'][:12]}"
        )
    elif args.command == "observables":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count or args.workers < 1:
            raise ValueError("Workers/partes positivos e índice válido.")
        planned = observables.plan(
            args.output, campaign, allow_partial=args.allow_partial, refine=args.refine
        )
        print(
            f"Observáveis: {len(planned['points'])} pontos totais; {len(planned['skipped_columns'])} colunas sem faixa utilizável."
        )
    else:
        if args.points_file:
            with args.points_file.open(encoding="utf-8-sig", newline="") as stream:
                requests = [
                    Parameters(float(r["m6_GeV"]), float(r["m8_GeV"]), float(r["C"]))
                    for r in csv.DictReader(stream)
                ]
        elif args.C is not None:
            requests = [Parameters(args.m6, args.m8, args.C)]
        else:
            raise ValueError("point exige --C ou --points-file.")
        if any(p.m8_GeV not in campaign.m8_values for p in requests):
            raise ValueError("m8 do ponto deve pertencer aos cenários do manifesto.")
        print(f"Pontos específicos: {len(requests)}")
    if args.dry_run:
        print("Configuração compatível; nenhum cálculo novo.")
        return
    budget = Budget(args.max_points, args.wall_seconds)
    budget.install_signals()
    mode = "nucleation" if args.command == "boundaries" else "observables"
    evaluator = Evaluator(
        args.output,
        campaign.numerics,
        mode,
        budget,
        timeout=args.timeout,
        scratch=args.scratch,
        retry_failed=args.retry_failed,
    )
    lock_name = (
        f"point-{point_id(requests[0])}"
        if args.command == "point" and requests
        else f"{args.command}-{getattr(args, 'shard_index', 0)}-of-{getattr(args, 'shard_count', 1)}"
    )
    with task_lock(args.output, lock_name):
        try:
            if args.command == "boundaries":
                boundaries.run(
                    args.output,
                    campaign,
                    evaluator,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    retry_failed=args.retry_failed,
                )
            elif args.command == "observables":
                observables.run(
                    args.output,
                    planned,
                    evaluator,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    workers=args.workers,
                )
            else:
                for parameters in requests:
                    evaluator(parameters)
        except Paused as message:
            print(message)
        finally:
            # Somente este estágio escreve seu próprio snapshot. A exportação
            # global é um comando posterior, depois que o array terminou.
            atomic_json(
                Path(args.output) / "progress" / f"{lock_name}.json",
                {"new_points": len(evaluator.evaluated), "paused": budget.stopping},
            )


def main(argv=None):
    """Despacha os cinco estágios; erros de configuração têm saída curta e código 2."""
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "init":
            init_campaign(args)
        elif args.command in ("boundaries", "observables", "point"):
            execute(args)
        elif args.command == "status":
            status(args.output)
        elif args.command == "check-pilot":
            campaign, manifest = load_campaign(args.output)
            records = point_records(args.output, "observables")
            missing = [
                case
                for case in campaign.m8_values
                if not any(
                    r["parameters"]["m8_GeV"] == safe(case)
                    and r["summary"].get("observables_complete")
                    and r["summary"]["decision"] == "yes"
                    for r in records
                )
            ]
            if missing:
                raise ValueError(
                    f"Piloto ainda sem observáveis completos nos cenários {safe(missing)}. Veja os JSONs dos pontos."
                )
            print(
                "Piloto aprovado: nucleação aceita e observáveis completos em todos os cenários. "
                "Isto valida o fluxo; convergência e aproximações físicas estão documentadas separadamente."
            )
        elif args.command == "figures":
            from .figures import render

            render(args)
        else:
            load_campaign(args.output, verify=False)
            print(export_tables(args.output))
            if args.zip:
                destination, source = args.zip.resolve(), args.output.resolve()
                if destination == source or source in destination.parents:
                    raise ValueError("O ZIP deve ficar fora da pasta que será compactada.")
                if destination.suffix.lower() != ".zip":
                    raise ValueError("Use um nome terminado em .zip.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                archive = archive_campaign(source, destination)
                print(f"ZIP completo: {archive}")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
