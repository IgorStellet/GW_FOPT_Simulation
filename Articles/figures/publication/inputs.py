"""União dos arquivos recebidos, preservando procedência e campos ausentes."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from contextlib import contextmanager
from pathlib import Path

from Articles.figures.data import (
    Dataset,
    coordinate,
    detector_metrics,
    finite,
    gw_valid,
)
from Articles.figures.frontiers.data import (
    beta_valid,
    load_exports,
    load_sources,
    read_table,
)


@contextmanager
def table_folder(source, names):
    """Abre pasta ou ZIP diretamente; não extrai nem cria temporários."""
    source = Path(source).resolve()
    if source.is_dir():
        yield source
        return
    with zipfile.ZipFile(source) as archive:
        yield zipfile.Path(archive)


def source_hash(source):
    digest = hashlib.sha256()
    with Path(source).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(rows, label):
    """Coordenadas repetidas não podem ser silenciosamente sobrepostas."""
    result = {}
    for row in rows:
        key = coordinate(row)
        if key in result:
            raise ValueError(f"Coordenada duplicada em {label}: {key}")
        result[key] = row
    return result


def add_peak_metrics(row, frequency, amplitude, years):
    """Métricas algébricas dos detectores a partir de um pico já calculado.

    Isso não recupera alpha/beta de um espectro. As funções PIS são as mesmas
    do núcleo; sua entrada é mHz e os arquivos de análise guardam Hz.
    """
    if not (
        finite(frequency) and finite(amplitude) and frequency > 0 and amplitude > 0
    ):
        return
    from Articles.core_imports import (
        bbo_sensitivity_s_pis,
        decigo_sensitivity_s_pis,
        lisa_sensitivity_s_pis,
    )

    row.update(f_sw_peak_Hz=frequency, omega_sw_peak_h2=amplitude)
    for name, sensitivity in zip(
        ("LISA", "DECIGO", "BBO"),
        (lisa_sensitivity_s_pis, decigo_sensitivity_s_pis, bbo_sensitivity_s_pis),
        strict=True,
    ):
        row[f"{name}_rho_sw"] = (
            math.sqrt(years) * amplitude / float(sensitivity(frequency * 1000))
        )


def load_received(exports, frontiers, full_table=None, *, years=4, threshold=10):
    """Usa o CSV completo v2 quando recebido, sem misturar transições antigas.

    Os ZIPs de revisão v1 continham observáveis somente de bordas/benchmarks.
    Na ausência do CSV completo, a leitura funciona como prévia explicitamente
    incompleta: não inventa os valores interiores omitidos da transferência.
    """
    old_names = (
        "figure_report.json",
        "points_analysis.csv",
        "boundary_brackets.csv",
        "all_saved_transitions.csv",
    )
    with table_folder(exports, old_names) as folder:
        old = load_exports(folder, years)
    new_names = (
        "frontier_report.json",
        "edge_observables.csv",
        "lisa_profiles.json",
        "lisa_endpoint_benchmarks.csv",
        "strength_reference.csv",
    )
    with table_folder(frontiers, new_names) as folder:
        report = json.loads(
            (folder / "frontier_report.json").read_text(encoding="utf-8")
        )
        edges = read_table(folder / "edge_observables.csv")
        benchmarks = read_table(folder / "lisa_endpoint_benchmarks.csv")
        profiles = json.loads(
            (folder / "lisa_profiles.json").read_text(encoding="utf-8")
        )
    expected = report.get("source_audit", {}).get("points")
    if full_table:
        full_table = Path(full_table)
        if full_table.suffix.lower() == ".csv":
            rows = read_table(full_table)
        else:
            with table_folder(
                full_table,
                (
                    "all_points_analysis.csv",
                    "points_analysis.csv",
                    "observables_complete.csv",
                ),
            ) as folder:
                name = next(
                    (
                        name
                        for name in (
                            "all_points_analysis.csv",
                            "points_analysis.csv",
                            "observables_complete.csv",
                        )
                        if (folder / name).exists()
                    ),
                    "all_points_analysis.csv",
                )
                rows = read_table(folder / name)
        merged = _unique(rows, "tabela completa")
        if expected is not None and len(rows) != expected:
            raise ValueError(
                f"A tabela tem {len(rows)} pontos; o relatório v2 registra {expected}."
            )
        for witness in edges:
            found = merged.get(coordinate(witness))
            if found is None or found["point_id"] != witness["point_id"]:
                raise ValueError(
                    "Tabela completa e bordas recebidas pertencem a campanhas diferentes."
                )
            for field in ("Tn_GeV", "alpha_trace", "beta_over_H"):
                if field == "beta_over_H" and not beta_valid(witness):
                    continue  # Um suplemento pode resolver a derivada antes rejeitada.
                a, b = witness.get(field), found.get(field)
                if (
                    finite(a)
                    and finite(b)
                    and not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-9)
                ):
                    raise ValueError(
                        f"Resultado divergente para {field} em {coordinate(witness)}."
                    )
    else:
        merged = _unique(old.rows, "primeira leva")
        for row in [*edges, *benchmarks]:
            # Substitui o registro físico inteiro: não combina alpha de uma
            # transição com beta/Tn de outra avaliação.
            merged[coordinate(row)] = dict(row)
    audit = {
        **report["source_audit"],
        "received_frontier_report": report,
        "complete_point_table_received": bool(full_table),
        "received_input_hashes": {},
    }
    for source in (exports, frontiers, full_table):
        if source is not None and Path(source).is_file():
            audit["received_input_hashes"][str(Path(source).resolve())] = source_hash(
                source
            )
    # Perfis fornecem SNR/pico, mas não os observáveis termodinâmicos omitidos.
    # Registros completos continuam sendo a autoridade para os mapas.
    for profile in profiles:
        case, mass = float(profile["m8_GeV"]), float(profile["m6_GeV"])
        for sample in profile["samples"]:
            key = case, mass, round(float(sample["C"]), 12)
            row = merged.get(key)
            if row is None:
                if full_table:
                    raise ValueError(
                        f"Perfil sem ponto correspondente na tabela completa: {key}."
                    )
                row = {
                    "m8_GeV": case,
                    "m6_GeV": mass,
                    "C": sample["C"],
                    "point_id": sample["point_id"],
                    "boundary_decision": sample.get("strong_decision", "unknown"),
                    "status": sample.get("point_status", "unknown"),
                    "thermodynamics_not_transferred": True,
                }
                merged[key] = row
            if row["point_id"] != sample["point_id"]:
                raise ValueError(
                    "Perfil e tabela têm identidades de ponto incompatíveis."
                )
            if not full_table and (
                sample.get("reason") == "resolved_acoustic_SNR"
                or (
                    sample.get("rho_sw") is not None
                    and sample.get("f_sw_peak_Hz") is not None
                )
            ):
                add_peak_metrics(
                    row,
                    sample.get("f_sw_peak_Hz"),
                    sample.get("omega_sw_peak_h2"),
                    years,
                )
    for row in merged.values():
        if not row.get("source_conflict"):
            if full_table and gw_valid(row):
                metrics = detector_metrics(
                    row, row.get("spectrum_defaults") or {}, years
                )
                for field in ("f_sw_peak_Hz", "omega_sw_peak_h2"):
                    if finite(row.get(field)) and not math.isclose(
                        row[field], metrics[field], rel_tol=1e-7, abs_tol=0
                    ):
                        raise ValueError(
                            f"Pico salvo incompatível com o núcleo GW atual: {row['point_id']}, {field}."
                        )
                row.update(metrics)
            else:
                add_peak_metrics(
                    row, row.get("f_sw_peak_Hz"), row.get("omega_sw_peak_h2"), years
                )
    dataset = Dataset(list(merged.values()), old.transitions, old.columns, [], audit)
    # Os arquivos podem ser reanalisados para outro limiar/tempo. Nunca reutiliza
    # diretamente intervalos certificados em uma configuração diferente.
    audit["original_detector_configuration"] = report.get("assumptions", {})
    audit["publication_detector_configuration"] = {
        "years": years,
        "threshold": threshold,
    }
    _refresh_profile_decisions(profiles, dataset.rows, threshold)
    return dataset, profiles


def _refresh_profile_decisions(profiles, rows, threshold):
    """Reclassifica os cortes com a união atual e o tempo/limiar escolhidos."""
    by_key = {coordinate(row): row for row in rows}
    for profile in profiles:
        for sample in profile["samples"]:
            key = (
                float(profile["m8_GeV"]),
                float(profile["m6_GeV"]),
                round(sample["C"], 12),
            )
            row = by_key.get(key, {})
            rho = row.get("LISA_rho_sw")
            if finite(rho) and not row.get("source_conflict"):
                sample.update(rho_sw=rho, decision="yes" if rho >= threshold else "no")
            elif sample.get("reason") == "no_accepted_nucleation_found":
                sample.update(rho_sw=None, decision="no")
            else:
                sample.update(rho_sw=None, decision="unknown")


def add_half_step_comparison(dataset, *, years=4, threshold=10):
    """Reutiliza beta(h/2) salvo, sem novas derivadas: diagnóstico, não erro estatístico."""
    for row in dataset.rows:
        half = row.get("beta_over_H_half_step")
        if (
            row.get("source_conflict")
            or not gw_valid(row)
            or not finite(half)
            or half <= 0
        ):
            continue
        alternative = detector_metrics(
            {**row, "beta_over_H": half}, row.get("spectrum_defaults") or {}, years
        )
        row["f_sw_peak_Hz_beta_half_step"] = alternative["f_sw_peak_Hz"]
        row["omega_sw_peak_h2_beta_half_step"] = alternative["omega_sw_peak_h2"]
        for detector in ("LISA", "DECIGO", "BBO"):
            field = detector + "_rho_sw"
            row[field + "_half_step"] = alternative[field]
            row[detector + "_step_classification_agrees"] = (
                row[field] >= threshold
            ) == (alternative[field] >= threshold)
            row[detector + "_detectable_at_both_steps"] = (
                min(row[field], alternative[field]) >= threshold
            )


def load_original(boundaries, observables=(), reach=(), *, years=4, threshold=10):
    """Alternativa no CHE: lê diretamente os bancos, com o leitor auditado v4."""
    from types import SimpleNamespace

    from Articles.figures.frontiers.render import reach_profiles

    dataset = load_sources(boundaries, [*observables, *reach], years=years)
    # O JSON dos bancos representa uma massa desligada como a string "inf".
    # CSVs restauram float(inf) em read_table; a leitura direta precisa oferecer
    # a mesma representação às comparações, tabelas e chaves de cores. Apenas
    # a visão em memória é normalizada: os bancos e manifestos ficam intactos.
    for row in dataset.rows:
        case, mass, _ = coordinate(row)
        row.update(m8_GeV=case, m6_GeV=mass)
    saved_configuration = {}
    for root in reach:
        for path in sorted(Path(root).glob("m8-*/interior_state.json")):
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("kind") == "lisa_cuts":
                saved_configuration = saved
                break
        if saved_configuration:
            break
    args = SimpleNamespace(
        observation_years=saved_configuration.get("observation_years", years),
        snr_threshold=saved_configuration.get("snr_threshold", threshold),
        C_tolerance=0.01,
        maximum_C_gap=0.0500001,
    )
    profiles = list(reach_profiles(dataset, reach, args).values())
    _refresh_profile_decisions(profiles, dataset.rows, threshold)
    dataset.audit["complete_point_table_received"] = True
    return dataset, profiles
