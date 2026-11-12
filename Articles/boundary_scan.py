"""Duas etapas independentes: fronteiras baratas, depois observáveis no interior."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import signal
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path

from Articles.adaptive_scan import nucleation_decision
from Articles.boundary_physics import classify
from Articles.boundary_search import (
    BoundaryPaused,
    BoundaryPlan,
    interior_coordinates,
    trace_boundaries,
)
from Articles.bounded_evaluation import evaluate_bounded
from Articles.cluster_runner import copy_file_atomic, restore_campaign
from Articles.collect_data import (
    M8_SCENARIOS_GEV,
    ROOT,
    Settings,
    export_csv,
    json_safe,
    open_database,
    point_id,
    provenance,
    save_point,
    single_writer,
    validate_campaign,
)
from Articles.combined_model import ModelParameters


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def reuse_compatible(source_manifest, target_manifest, stage):
    """Reuso explícito exige mesmo núcleo/modelo, versões e escolhas físicas.

    O coletor/estratégia pode mudar. Cada registro conserva a origem, e a
    aceitação física é conferida de novo ao classificar a região selecionada.
    """
    old, new = source_manifest.get("source_sha256", {}), target_manifest["source_sha256"]
    # Compara o pacote de núcleo EFETIVAMENTE importado. O arquivo de aliases
    # core_imports.py não implementa física e pode ter LF/CRLF distintos.
    required = [name for name in new if name.startswith("src/CosmoTransitions/")
                or name == "Articles/combined_model.py"]
    if any(old.get(name) != new[name] for name in required):
        raise ValueError("Reuso incompatível: núcleo/modelo diferente ou origem sem hashes.")
    if source_manifest.get("versions") != target_manifest["versions"]:
        raise ValueError("Reuso incompatível: versões de NumPy/SciPy diferentes.")
    ignored = {"recovery_attempts"}
    if stage == "boundaries":
        ignored |= {"beta_step", "beta_check", "beta_relative_tolerance", "g_star", "wall_velocity", "strong_threshold"}
    old_settings = source_manifest.get("settings", {})
    for key, value in target_manifest["settings"].items():
        if key not in ignored and old_settings.get(key) != value:
            raise ValueError(f"Reuso incompatível: configuração {key} diferente.")


def find_slice(root, scenario):
    root = Path(root)
    return root if (root / "scan.sqlite").exists() else root / f"m8-{scenario}"


def source_manifest(folder):
    if not (folder / "scan.sqlite").exists():
        raise ValueError(f"Campanha para reuso não encontrada: {folder}")
    path = folder / "manifest.json"
    if not path.exists():
        raise ValueError(f"Origem sem manifesto: {folder}")
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def readonly_source_snapshot(folder, scratch):
    """Copia DB+WAL por leitura de arquivos; não cria locks/SHM na origem.

    SQLite mode=ro ainda pode precisar criar sidecars quando abre um WAL.
    Abrimos apenas a cópia local. A origem deve estar parada; se mudar durante
    a cópia, rejeitamos a importação em vez de misturar instantes diferentes.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    paths = [folder / "scan.sqlite", folder / "scan.sqlite-wal"]
    def signatures():
        return [(p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths]
    with tempfile.TemporaryDirectory(prefix="reuse-", dir=scratch) as temporary:
        snapshot = Path(temporary)
        before = signatures()
        for path in paths:
            if path.exists():
                shutil.copyfile(path, snapshot / path.name)
        if signatures() != before:
            raise ValueError("Origem mudou durante o reuso. Pare a campanha de origem e tente novamente.")
        with closing(sqlite3.connect(snapshot / "scan.sqlite")) as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Snapshot da origem não íntegro; nenhum dado dessa origem foi importado.")
            yield db


class BoundaryStore:
    """SQLite sempre local ao nó; publicação atômica por ponto quando solicitada."""

    def __init__(self, work, output, manifest, settings, plan, stage, timeout, max_points=None, retry_failed=False):
        self.work, self.output = Path(work), Path(output)
        self.manifest, self.settings, self.plan = manifest, settings, plan
        self.stage, self.timeout, self.max_points = stage, timeout, max_points
        self.retry_failed, self.retried = retry_failed, set()
        self.new_points = 0
        self.stop_requested = False
        self.cache = {}
        self.imported_masses = set()
        self._revision = 0
        self._published_revision = 0 if (self.output / "scan.sqlite").exists() else None
        self.allowed_coordinates = None
        with closing(open_database(self.work, manifest)) as db:
            transitions = {}
            for key, record in db.execute("SELECT point_id, record FROM transitions ORDER BY transition_index"):
                transitions.setdefault(key, []).append(json.loads(record))
            for key, record in db.execute("SELECT point_id, record FROM points"):
                self.cache[key] = (json.loads(record), transitions.get(key, []))

    def publish(self, new_details=None):
        if self.work == self.output:
            return
        if new_details:
            copy_file_atomic(self.work / new_details, self.output / new_details)
        if (self.work / "manifest.json").exists():
            copy_file_atomic(self.work / "manifest.json", self.output / "manifest.json")
        # A cópia publicada nunca é um WAL ativo no filesystem compartilhado.
        if self._published_revision != self._revision:
            snapshot = self.work / "publish.sqlite.tmp"
            try:
                with closing(sqlite3.connect(self.work / "scan.sqlite")) as source, closing(sqlite3.connect(snapshot)) as target:
                    source.backup(target)
                    target.execute("PRAGMA journal_mode=DELETE")
                copy_file_atomic(snapshot, self.output / "scan.sqlite")
                self._published_revision = self._revision
            finally:
                snapshot.unlink(missing_ok=True)
        # Publica o estado DEPOIS dos pontos que o sustentam, inclusive após
        # uma interrupção entre as substituições atômicas dos arquivos.
        for name in ("boundary_state.json", "interior_state.json", "boundaries.csv", "points.csv", "transitions.csv"):
            if (self.work / name).exists():
                copy_file_atomic(self.work / name, self.output / name)

    def save(self, row, transitions, details):
        decision, reason = classify(row, transitions, self.plan.criterion, self.plan.strong_threshold)
        row.update(boundary_decision=decision, selection_reason=reason,
                   scan_decision=nucleation_decision(row, transitions))
        with closing(open_database(self.work, self.manifest)) as db:
            save_point(db, self.work, (row, transitions, details))
        self.cache[row["point_id"]] = row, transitions
        self._revision += 1
        self.publish(row["details_path"])

    def evaluate(self, C, mass):
        if (self.stop_requested or (self.output / "STOP").exists()
                or (self.output.parent / "STOP").exists()):
            raise BoundaryPaused("Parada solicitada; todos os pontos concluídos foram salvos.")
        params = ModelParameters(C=C, m6_GeV=float(mass), m8_GeV=self.plan.m8_GeV)
        key = point_id(params)
        old = self.cache.get(key)
        retry = old and self.retry_failed and old[0].get("boundary_decision") == "unknown" and key not in self.retried
        repair = old is None or not (self.work / old[0].get("details_path", "missing")).is_file()
        if repair or retry:
            if self.max_points is not None and self.new_points >= self.max_points:
                raise BoundaryPaused("Orçamento de pontos novos atingido; retome com o mesmo comando.")
            result = evaluate_bounded(params, self.settings, timeout=self.timeout, mode=self.stage,
                                      scratch=self.work / ".point-tmp")
            self.save(*result)
            self.retried.add(key)
            self.new_points += 1
            row = result[0]
            print(f"[{self.new_points}] m8={self.plan.m8_GeV:g}, m6={float(mass):g}, C={C:g}: "
                  f"{row['status']} / região={row['boundary_decision']} ({row['runtime_seconds']:.1f}s)", flush=True)
        row = self.cache[key][0]
        return {"decision": row["boundary_decision"], "reason": row["selection_reason"], "status": row["status"]}

    def hints(self, mass):
        return [{"C": row["C"], "decision": row["boundary_decision"]} for row, _ in self.cache.values()
                if float(row["m6_GeV"]) == mass]

    def import_campaign(self, folder):
        """A origem fica somente de leitura; desconhecidos não entram no cache novo."""
        origin = source_manifest(folder)
        reuse_compatible(origin, self.manifest, self.stage)
        count = 0
        unknown_count = 0
        with readonly_source_snapshot(folder, self.work / ".reuse-tmp") as db:
            db.execute("BEGIN")
            saved = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
            if saved["fingerprint"] != origin["fingerprint"]:
                raise ValueError("Manifesto JSON difere do banco de origem; reuso recusado.")
            for key, record, details_path in db.execute("SELECT point_id,record,details_path FROM points"):
                if key in self.cache:
                    continue
                row = json.loads(record)
                if float(row["m8_GeV"]) != self.plan.m8_GeV:
                    continue
                masses = self.plan.masses()
                if self.plan.column_limit:
                    masses = masses[:self.plan.column_limit]
                if float(row["m6_GeV"]) not in masses or not self.plan.C_bounds[0] <= row["C"] <= self.plan.C_bounds[1]:
                    continue
                if self.allowed_coordinates is not None and (row["C"], float(row["m6_GeV"])) not in self.allowed_coordinates:
                    continue
                transitions = [json.loads(r[0]) for r in db.execute(
                    "SELECT record FROM transitions WHERE point_id=? ORDER BY transition_index", (key,))]
                decision, _ = classify(row, transitions, self.plan.criterion, self.plan.strong_threshold)
                if decision == "unknown":
                    same_boundary_budget = (self.stage == "boundaries" and origin.get("grid", {}).get("stage") == "boundaries"
                                            and origin["grid"].get("point_timeout_seconds") == self.timeout)
                    if not same_boundary_budget:
                        continue
                if self.stage == "interior" and decision == "yes" and not any(
                    t.get("status") == "nucleated" and not t.get("quality_flags")
                    and all(t.get(k) is not None for k in ("alpha_trace", "beta_over_H", "Tn_GeV")) for t in transitions
                ):
                    continue
                detail = (folder / details_path).resolve()
                if not detail.is_relative_to(folder.resolve()) or not detail.is_file():
                    continue
                with gzip.open(detail, "rt", encoding="utf-8") as stream:
                    diagnostics = json.load(stream)
                row.update(imported_from=str(folder.resolve()), imported_fingerprint=origin["fingerprint"])
                if self.stage == "boundaries":
                    row["imported_original_status"] = row["status"]
                    row["calculation_mode"] = "boundaries_reused"
                    if any(t.get("nucleation_accepted") for t in transitions):
                        row["status"] = "nucleated"
                self.save(row, transitions, diagnostics)
                self.imported_masses.add(float(row["m6_GeV"]))
                count += 1
                unknown_count += decision == "unknown"
        print(f"Reuso: {count} registros de {folder}; {unknown_count} inconclusivos preservados; origem intacta.", flush=True)
        return count

    def export(self):
        with closing(open_database(self.work, self.manifest)) as db:
            export_csv(db, self.work)


def export_boundaries(folder, report):
    import csv

    folder = Path(folder)
    atomic_json(folder / "boundary_state.json", report)
    keys = ("m6_GeV", "state", "seed_C", "lower_inside_C", "lower_outside_C", "lower_width_C",
            "lower_state", "lower_outside_kind", "upper_inside_C", "upper_outside_C", "upper_width_C",
            "upper_state", "upper_outside_kind")
    temporary = folder / "boundaries.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for column in report["columns"]:
            row = {k: json_safe(column[k]) for k in ("m6_GeV", "state", "seed_C")}
            for side in ("lower", "upper"):
                edge = column.get(side) or {}
                for key in ("inside_C", "outside_C", "width_C", "state", "outside_kind"):
                    row[f"{side}_{key}"] = edge.get(key)
            writer.writerow(row)
    temporary.replace(folder / "boundaries.csv")


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--work-dir", type=Path)
    result.add_argument("--stage", choices=("boundaries", "interior"), default="boundaries")
    result.add_argument("--scenario-index", type=int, choices=range(3), default=0)
    result.add_argument("--criterion", choices=("strong", "nucleation"), default="strong")
    result.add_argument("--strong-threshold", type=float, default=1.0)
    result.add_argument("--point-timeout", type=float, default=300.0)
    result.add_argument("--C-bounds", type=float, nargs=2, default=(0, 10))
    result.add_argument("--C-tolerance", type=float, default=0.02)
    result.add_argument("--walk-step", type=float, default=0.1)
    result.add_argument("--seed-step", type=float, default=0.1)
    result.add_argument("--seed-resolution", type=float, default=0.02)
    result.add_argument("--seed-C", type=float, default=3.83)
    result.add_argument("--m6", type=float, nargs=3, default=(500, 2000, 5))
    result.add_argument("--patience", type=int, default=3)
    result.add_argument("--unknown-limit", type=int, default=3)
    result.add_argument("--smoke", action="store_true", help="Duas massas; localiza as fronteiras reais nessas colunas.")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--plots", action="store_true")
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--max-points", type=int)
    result.add_argument("--reuse", type=Path, action="append", default=[])
    result.add_argument("--boundaries", type=Path, help="Campanha de fronteiras usada na etapa interior.")
    result.add_argument("--interior-step", type=float, default=0.02)
    result.add_argument("--allow-partial-boundaries", action="store_true")
    result.add_argument("--beta-check", action=argparse.BooleanOptionalAction, default=True,
                        help="Somente na etapa interior; nunca calcula beta na etapa fronteiras.")
    return result


def prepare(args):
    args.output = (ROOT / args.output).resolve()
    if args.work_dir:
        args.work_dir = (ROOT / args.work_dir).resolve()
        if (args.work_dir == args.output or args.work_dir in args.output.parents or args.output in args.work_dir.parents):
            raise ValueError("work-dir e output devem ser distintos e não aninhados.")
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            raise ValueError("work-dir deve ser uma pasta local nova/vazia.")
    if not math.isfinite(args.point_timeout) or args.point_timeout <= 0:
        raise ValueError("point-timeout deve ser finito e positivo.")
    if args.max_points is not None and args.max_points < 1:
        raise ValueError("max-points deve ser positivo.")
    if not math.isfinite(args.interior_step) or args.interior_step <= 0:
        raise ValueError("interior-step deve ser finito e positivo.")
    plan = BoundaryPlan(m6_range_GeV=tuple(args.m6), C_bounds=tuple(args.C_bounds),
                        m8_GeV=M8_SCENARIOS_GEV[args.scenario_index], seed_C=args.seed_C,
                        C_tolerance=args.C_tolerance, walk_step=args.walk_step, seed_step=args.seed_step,
                        seed_resolution=args.seed_resolution, patience=args.patience, unknown_limit=args.unknown_limit,
                        criterion=args.criterion, strong_threshold=args.strong_threshold, column_limit=2 if args.smoke else None)
    plan.validate()
    settings = Settings(recovery_attempts=0, beta_check=args.beta_check, strong_threshold=args.strong_threshold)
    settings.validate()
    grid = {**plan.identity(), "stage": args.stage, "point_timeout_seconds": args.point_timeout}
    boundary_report = boundary_origin = None
    if args.stage == "interior":
        if not args.boundaries:
            raise ValueError("A etapa interior exige --boundaries.")
        folder = (ROOT / args.boundaries).resolve()
        folder = folder if (folder / "boundary_state.json").exists() else folder / f"m8-{args.scenario_index}"
        path = folder / "boundary_state.json"
        boundary_bytes = path.read_bytes()
        boundary_report = json.loads(boundary_bytes)
        saved_plan = boundary_report["plan"]
        if float(saved_plan["m8_GeV"]) != plan.m8_GeV or saved_plan["criterion"] != plan.criterion or saved_plan["strong_threshold"] != plan.strong_threshold:
            raise ValueError("Cenário/critério/corte difere da campanha de fronteiras.")
        valid_states = ("pilot_complete", "complete") if args.smoke else ("complete",)
        if not args.allow_partial_boundaries and boundary_report["state"] not in valid_states:
            raise ValueError("Fronteiras parciais/inconclusivas: revise ou use --allow-partial-boundaries para somente colunas bounded.")
        boundary_origin = source_manifest(folder)
        grid.update(boundary_fingerprint=boundary_origin["fingerprint"],
                    boundary_report_sha256=hashlib.sha256(boundary_bytes).hexdigest(),
                    interior_step=args.interior_step, interior_smoke=args.smoke)
        grid["allow_partial_boundaries"] = args.allow_partial_boundaries
    manifest = provenance(settings, grid)
    if boundary_origin:
        reuse_compatible(boundary_origin, manifest, "boundaries")
    validate_campaign(args.output, manifest)
    for source in args.reuse:
        folder = find_slice((ROOT / source).resolve(), args.scenario_index)
        if folder == args.output or args.output in folder.parents or folder in args.output.parents:
            raise ValueError("Origem de reuso e destino devem ser distintos e não aninhados.")
        reuse_compatible(source_manifest(folder), manifest, args.stage)
    return plan, settings, manifest, boundary_report


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        plan, settings, manifest, boundary_report = prepare(args)
    except (ValueError, OSError, KeyError) as error:
        arguments.error(str(error))
    print(f"Etapa {args.stage}; m8={plan.m8_GeV:g}; critério={plan.criterion}; "
          f"orçamento={args.point_timeout:g}s/ponto; destino={args.output}", flush=True)
    if args.dry_run:
        print("Configuração compatível; nenhum cálculo/pasta nova.", flush=True)
        return 0
    work = args.work_dir or args.output
    args.output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    report = {}
    with single_writer(args.output):
        if work != args.output:
            restore_campaign(args.output, work)
        store = BoundaryStore(work, args.output, manifest, settings, plan, args.stage, args.point_timeout,
                              args.max_points, args.retry_failed)
        if args.stage == "interior":
            store.allowed_coordinates = set(interior_coordinates(boundary_report, args.interior_step, 2 if args.smoke else None,
                                                                  args.allow_partial_boundaries))
        previous_handlers = {}
        def request_stop(signum, frame):
            store.stop_requested = True
            print("Parada solicitada: publicar após o ponto atual (limitado pelo orçamento).", flush=True)
        for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous_handlers[number] = signal.signal(number, request_stop)
        last_column_count = -1

        def checkpoint(current):
            nonlocal last_column_count
            if args.stage == "boundaries":
                export_boundaries(work, current)
                if len(current["columns"]) != last_column_count or current["state"] != "running":
                    store.export()
                    last_column_count = len(current["columns"])
            else:
                atomic_json(work / "interior_state.json", current)
            store.publish()

        try:
            for source in args.reuse:
                store.import_campaign(find_slice((ROOT / source).resolve(), args.scenario_index))
            if args.stage == "boundaries":
                saved_path = work / "boundary_state.json"
                prior_columns = json.loads(saved_path.read_text(encoding="utf-8")).get("columns", []) if saved_path.exists() else []
                prior_columns = [c for c in prior_columns if float(c["m6_GeV"]) not in store.imported_masses]
                trace_boundaries(plan, store.evaluate, report, checkpoint, store.hints,
                                 None if args.retry_failed else prior_columns)
                recorded = {(p["C"], float(p["m6_GeV"])) for p in report["unknown_points"]}
                for row, _ in store.cache.values():
                    if row["boundary_decision"] == "unknown" and (row["C"], float(row["m6_GeV"])) not in recorded:
                        report["unknown_points"].append({"C": row["C"], "m6_GeV": row["m6_GeV"],
                                                         "decision": "unknown", "reason": row["selection_reason"]})
                if report["unknown_points"] and report["state"] in ("complete", "pilot_complete"):
                    report["state"] += "_with_unresolved"
                checkpoint(report)
            else:
                coordinates = list(interior_coordinates(boundary_report, args.interior_step, 2 if args.smoke else None,
                                                        args.allow_partial_boundaries))
                used_masses = {float(mass) for _, mass in coordinates}
                report.update(state="running", points_planned=0, points_checked=0, unknown_points=[], selection_misses=[],
                              skipped_columns=[c["m6_GeV"] for c in boundary_report["columns"]
                                               if c["state"] != "no_selected_seed_on_grid" and float(c["m6_GeV"]) not in used_masses],
                              partial_columns=[c["m6_GeV"] for c in boundary_report["columns"]
                                               if c["state"] != "bounded" and float(c["m6_GeV"]) in used_masses],
                              excluded_columns=[c["m6_GeV"] for c in boundary_report["columns"] if c["state"] == "no_selected_seed_on_grid"])
                report["points_planned"] = len(coordinates)
                try:
                    for C, mass in coordinates:
                        result = store.evaluate(C, mass)
                        report["points_checked"] += 1
                        if result["decision"] == "unknown" or result.get("status") == "observables_unresolved":
                            report["unknown_points"].append({"C": C, "m6_GeV": mass, **result})
                        if result["decision"] == "no":
                            report["selection_misses"].append({"C": C, "m6_GeV": mass, **result})
                        checkpoint(report)
                    report["state"] = "pilot_complete" if args.smoke else "complete"
                    if report["unknown_points"] or report["skipped_columns"] or report["partial_columns"] or report["selection_misses"]:
                        report["state"] += "_with_unresolved"
                except BoundaryPaused as error:
                    report.update(state="paused", message=str(error))
                checkpoint(report)
        except BaseException:
            report["state"] = "interrupted"
            if args.stage == "boundaries":
                report.setdefault("columns", [])
            checkpoint(report)
            raise
        finally:
            store.export()
            store.publish()
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
        if args.smoke or args.plots:
            if args.stage == "boundaries":
                from Articles.plot_boundaries import plot_boundary_campaign
                plot_boundary_campaign(args.output)
            else:
                from Articles.plot_results import plot_campaign
                plot_campaign(args.output)
    print(f"Estado: {report['state']}; {store.new_points} pontos novos; "
          f"{len(report.get('unknown_points', []))} inconclusivos.", flush=True)
    return 3 if report["state"].endswith("_with_unresolved") else 0


if __name__ == "__main__":
    raise SystemExit(main())
