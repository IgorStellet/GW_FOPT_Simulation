"""Geometria local, controles de qualidade e escolha explícita de benchmarks.

Interpolação é somente apresentação dos dados: nenhum observável físico é
calculado de novo aqui. As tolerâncias geométricas são exportadas no relatório.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear
from scipy.sparse import diags, vstack
from scipy.spatial import cKDTree

from Articles.figures.data import coordinate, finite, gw_valid
from Articles.figures.frontiers.data import (
    alpha_valid,
    beta_valid,
    edge_samples,
    merge_intervals,
)
from Articles.figures.geometry import adjacent_runs, supported_triangulation
from Articles.plot_results import reconstruct_spectrum

FIELDS = ("Tn_GeV", "alpha_trace", "beta_over_H", "vev_over_Tc")


def envelopes(dataset):
    """Extremos selecionados em C: não os confunde com extremos de alpha/beta."""
    grouped = defaultdict(dict)
    for row in edge_samples(dataset):
        grouped[coordinate(row)[:2]][row["edge"]] = row
    result = []
    for (case, mass), sides in sorted(grouped.items()):
        lower, upper = sides["lower"], sides["upper"]
        result.append(
            {
                "m8_GeV": case,
                "m6_GeV": mass,
                "lower_C": lower["C"],
                "upper_C": upper["C"],
                "lower_point_id": lower["point_id"],
                "upper_point_id": upper["point_id"],
                "lower_certified": lower["edge_certified"],
                "upper_certified": upper["edge_certified"],
                "interpretation": "envelope_of_selected_samples_not_proof_of_connected_interior",
                "lower_sample_C": lower["C"],
                "upper_sample_C": upper["C"],
                "lower_bound_C": lower.get("edge_outside_C")
                if lower.get("edge_certified")
                else 0.0,
                "upper_bound_C": upper.get("edge_outside_C")
                if upper.get("edge_certified")
                else math.inf,
            }
        )
    return result


def fit_boundary_guides(records, *, smoothness=1000):
    """Guia regularizada dentro dos limites efetivamente determinados.

    Uma testemunha interna só limita a borda de um lado; não é uma medição
    pontual da borda. Usa C_fora <= C_borda <= C_dentro na borda inferior e
    C_dentro <= C_borda <= C_fora na superior. Sem testemunha externa,
    permanece a restrição unilateral. A curvatura discreta é penalizada para
    não interpretar uma busca interrompida como um pico físico na fronteira.
    Todos os limites e deslocamentos da guia são exportados para auditoria.
    """
    fitted = [dict(row) for row in records]
    for case in {row["m8_GeV"] for row in fitted}:
        for run in adjacent_runs(
            [r for r in fitted if r["m8_GeV"] == case], maximum_gap=5.01
        ):
            n = len(run)
            if n < 3:
                continue
            roughness = diags(
                [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
                [0, 1, 2],
                shape=(n - 2, n),
                format="csr",
            )
            for side in ("lower", "upper"):
                samples = np.array([r[side + "_sample_C"] for r in run])
                outside = np.array([r[side + "_bound_C"] for r in run])
                if side == "lower":
                    low, high = outside, samples
                    bracketed = outside > 0
                else:
                    low, high = samples, outside
                    bracketed = np.isfinite(outside)
                width = np.where(bracketed, high - low, np.inf)
                weights = np.where(bracketed, 1 / np.maximum(width, 0.005), 0)
                midpoint = np.where(
                    bracketed,
                    (low + np.where(np.isfinite(high), high, low)) / 2,
                    samples,
                )
                matrix = vstack(
                    [diags(weights, format="csr"), smoothness * roughness], format="csr"
                )
                target = np.r_[weights * midpoint, np.zeros(n - 2)]
                high = np.maximum(high, low + 1e-11)
                solution = lsq_linear(
                    matrix.toarray(),
                    target,
                    bounds=(low, high),
                    method="bvls",
                    lsq_solver="exact",
                    tol=1e-9,
                    max_iter=1000,
                )
                if not solution.success:
                    raise ValueError(
                        f"Guia de borda sem convergência, m8={case}, {side}: {solution.message}"
                    )
                for row, value, sample in zip(run, solution.x, samples, strict=True):
                    row[side + "_C"] = float(value)
                    row[side + "_guide_minus_sample_C"] = float(value - sample)
                    row[side + "_guide_is_interpolated_estimate"] = True
                    row["guide_smoothness"] = smoothness
    return fitted


def envelope_curves(records, *, strength_axis=False, maximum_mass_gap=5.01):
    """PCHIP entre massas contíguas; sem overshoot, extrapolação ou filtro."""
    curves = []
    for run in adjacent_runs(records, maximum_gap=maximum_mass_gap):
        if len(run) < 2:
            continue
        masses = np.array([row["m6_GeV"] for row in run])
        x = (1000 / masses) ** 2 if strength_axis else masses
        order = np.argsort(x)
        x = x[order]
        low = np.array([row["lower_C"] for row in run])[order]
        high = np.array([row["upper_C"] for row in run])[order]
        dense = np.linspace(x[0], x[-1], max(30, len(run) * 4))
        a = PchipInterpolator(x, low, extrapolate=False)(dense)
        b = PchipInterpolator(x, high, extrapolate=False)(dense)
        # Interpolantes independentes de duas bordas podem cruzar. Não se
        # corrige a geometria trocando nomes: nesse trecho não há preenchimento.
        a[a > b] = np.nan
        curves.append((dense, a, b))
    return curves


def value_valid(row, field):
    if row.get("source_conflict"):
        return False
    if field == "alpha_trace":
        return alpha_valid(row)
    if field == "beta_over_H":
        return beta_valid(row)
    if "_rho_sw" in field:
        return finite(row.get(field)) and row[field] > 0
    if field == "vev_over_Tc":
        return (
            bool(row.get("nucleation_accepted"))
            and finite(row.get(field))
            and row[field] >= 0
        )
    return (
        bool(row.get("nucleation_accepted"))
        and finite(row.get(field))
        and row[field] > 0
    )


def field_mesh(rows, field, *, maximum_mass_gap=100.01, maximum_C_gap=0.081):
    """Triangulação em unidades da resolução, com nós inválidos preservados.

    Um cálculo boundary-only sem alpha não é uma avaliação fracassada de alpha.
    Ele não entra nessa malha. Uma avaliação adicional que falhou entra como
    nó inválido e mascara seus triângulos. Assim não se interpola através de
    uma derivada rejeitada, mas se pode interpolar entre cortes separados por
    100 GeV quando essa é a resolução REAL da coleta de observáveis.
    """
    nodes = []
    logarithmic = field in ("alpha_trace", "beta_over_H") or "_rho_sw" in field
    for row in rows:
        if not math.isfinite(float(row["m6_GeV"])):
            continue
        folder = str(row.get("source_folder", ""))
        additional = any(
            name in folder
            for name in ("edge_observables", "lisa_reach", "publication_refinement")
        )
        attempted = (
            value_valid(row, field)
            or additional
            or row.get("source_conflict")
            or row.get("boundary_decision") == "unknown"
        )
        if field == "beta_over_H":
            attempted |= finite(row.get("alpha_trace"))
        elif field == "alpha_trace":
            attempted |= row.get("alpha_trace") is not None
        elif field in ("Tn_GeV", "vev_over_Tc"):
            attempted = True
        elif "_rho_sw" in field:
            attempted |= finite(row.get("alpha_trace"))
        if not attempted:
            continue
        value = row.get(field) if value_valid(row, field) else None
        if value is not None and logarithmic:
            value = math.log10(value)
        nodes.append({**row, "plot_value": value})
    tri, values = supported_triangulation(
        nodes, "plot_value", max_mass_gap=maximum_mass_gap, max_C_gap=maximum_C_gap
    )
    triangles = len(tri.triangles) if tri is not None else 0
    supported = int(np.count_nonzero(~tri.mask)) if tri is not None else 0
    return (
        tri,
        values,
        {
            "field": field,
            "nodes": len(nodes),
            "valid_nodes": sum(value_valid(row, field) for row in nodes),
            "triangles": triangles,
            "supported_triangles": supported,
            "maximum_mass_gap_GeV": maximum_mass_gap,
            "maximum_C_gap": maximum_C_gap,
            "logarithmic_interpolation": logarithmic,
        },
    )


def availability(dataset, *, threshold=10):
    result = []
    for case in dataset.cases:
        rows = [row for row in dataset.rows if float(row["m8_GeV"]) == case]
        result.append(
            {
                "m8_GeV": case,
                "points": len(rows),
                "selected_points": sum(
                    row.get("boundary_decision") == "yes" for row in rows
                ),
                "alpha_valid": sum(alpha_valid(row) for row in rows),
                "beta_valid": sum(beta_valid(row) for row in rows),
                "GW_valid": sum(gw_valid(row) for row in rows),
                "LISA_detectable_both_steps": sum(
                    bool(row.get("LISA_detectable_at_both_steps")) for row in rows
                ),
                "LISA_step_classification_changes": sum(
                    row.get("LISA_step_classification_agrees") is False for row in rows
                ),
                "LISA_detectable": sum(
                    value_valid(row, "LISA_rho_sw") and row["LISA_rho_sw"] >= threshold
                    for row in rows
                ),
                "quality_flags": dict(
                    Counter(
                        str(row["quality_flags"])
                        for row in rows
                        if row.get("quality_flags")
                    )
                ),
            }
        )
    return result


def detector_cuts(
    dataset, profiles, *, threshold=10, maximum_C_gap=0.0500001, half_step=False
):
    """Tabela de intervalos locais, sem tornar uma falha distante veto global.

    Mantém singleton detectável para auditoria. Só há área/linha entre amostras
    consecutivas resolvidas. O limiar é interpolado em log(SNR), que também é
    a coordenada usada nos mapas. Fronteira física não é chamada de SNR=10.
    """
    rows = {coordinate(row): row for row in dataset.rows}
    cuts = []
    for profile in profiles:
        case, mass = float(profile["m8_GeV"]), float(profile["m6_GeV"])
        if not profile.get("grid_C"):
            continue  # Cortes planejados, não a malha auxiliar só de bordas.
        samples = sorted(profile["samples"], key=lambda item: item["C"])
        for detector in ("LISA", "DECIGO", "BBO"):
            field = detector + "_rho_sw" + ("_half_step" if half_step else "")
            for left, right in pairwise(samples):
                if right["C"] - left["C"] > maximum_C_gap:
                    continue
                a = rows.get((case, mass, round(left["C"], 12)), {})
                b = rows.get((case, mass, round(right["C"], 12)), {})
                if not all(value_valid(row, field) for row in (a, b)):
                    continue
                fa, fb = (
                    math.log10(a[field] / threshold),
                    math.log10(b[field] / threshold),
                )
                if max(fa, fb) < 0:
                    continue
                low, high = left["C"], right["C"]
                crossing = None
                if fa * fb < 0:
                    crossing = low + (high - low) * (-fa) / (fb - fa)
                    low, high = (crossing, high) if fa < 0 else (low, crossing)
                cuts.append(
                    {
                        "detector": detector,
                        "beta_stencil": "h/2" if half_step else "h",
                        "m8_GeV": case,
                        "m6_GeV": mass,
                        "C_lower": low,
                        "C_upper": high,
                        "threshold_C": crossing,
                        "left_point_id": a["point_id"],
                        "right_point_id": b["point_id"],
                        "C_bracket_width": right["C"] - left["C"],
                        "interpretation": "local_log_SNR_interpolation_between_resolved_samples",
                    }
                )
    return cuts


def match_spectra(dataset, *, threshold=10, maximum_peak_distance=0.16):
    """Busca dois trios de sinais próximos, sem otimizar uma conclusão física.

    Distância euclidiana em log10(f_peak), log10(Omega_peak). O segundo trio
    também exige massas distintas. A semelhança do espectro é medida depois,
    na banda em que o sinal acústico da referência supera 1% de seu pico.
    Não é likelihood instrumental nem prova de degenerescência estatística.
    """
    cases = dataset.cases
    if len(cases) != 3:
        return [], []
    pools = {
        case: sorted(
            [
                row
                for row in dataset.rows
                if float(row["m8_GeV"]) == case
                and math.isfinite(float(row["m6_GeV"]))
                and gw_valid(row)
                and value_valid(row, "LISA_rho_sw")
                and row["LISA_rho_sw"] >= threshold
                and value_valid(row, "LISA_rho_sw_half_step")
                and row["LISA_rho_sw_half_step"] >= threshold
            ],
            key=lambda row: (row["m6_GeV"], row["C"], row["point_id"]),
        )
        for case in cases
    }
    if any(not pool for pool in pools.values()):
        return [], []

    def features(records):
        return np.log10([[r["f_sw_peak_Hz"], r["omega_sw_peak_h2"]] for r in records])

    trees = {case: cKDTree(features(pools[case])) for case in cases[1:]}
    candidates = []
    for reference in pools[cases[0]]:
        partners, distances = [], []
        for case in cases[1:]:
            distance, index = trees[case].query(features([reference])[0])
            partners.append(pools[case][int(index)])
            distances.append(float(distance))
        records = [reference, *partners]
        if max(distances) <= maximum_peak_distance:
            candidates.append((max(distances), records))
    candidates.sort(key=lambda item: (item[0], item[1][0]["point_id"]))
    chosen = []
    # Primeiro busca uma comparação com a mesma massa nos três modelos.
    same_mass = []
    for mass in (1000.0, 1500.0):
        local = {
            case: [row for row in pools[case] if row["m6_GeV"] == mass]
            for case in cases
        }
        if any(not values for values in local.values()):
            continue
        for reference in local[cases[0]]:
            partners, distance = [], 0.0
            for case in cases[1:]:
                norms = np.linalg.norm(
                    features(local[case]) - features([reference])[0], axis=1
                )
                index = int(np.argmin(norms))
                partners.append(local[case][index])
                distance = max(distance, float(norms[index]))
            if distance <= maximum_peak_distance:
                same_mass.append((distance, [reference, *partners]))
    if same_mass:
        chosen.append(("A", min(same_mass, key=lambda item: item[0])[1]))
    varied = [
        item for item in candidates if len({row["m6_GeV"] for row in item[1]}) == 3
    ]
    if varied:
        chosen.append(("B", varied[0][1]))
    elif any(len({row["m6_GeV"] for row in item[1]}) > 1 for item in candidates):
        candidates = [
            item for item in candidates if len({row["m6_GeV"] for row in item[1]}) > 1
        ]
        alternative = next(
            (
                item
                for item in candidates
                if not chosen or item[1][0]["point_id"] != chosen[0][1][0]["point_id"]
            ),
            candidates[0],
        )
        chosen.append(("B", alternative[1]))
    exported, spectra_rows = [], []
    frequencies = np.geomspace(1e-5, 1, 900)
    for group, records in chosen:
        reference = records[0]
        reference_spectrum = reconstruct_spectrum(
            reference, frequencies, reference.get("spectrum_defaults") or {}
        )
        support = reference_spectrum["sw"] >= 0.01 * np.max(reference_spectrum["sw"])
        for index, row in enumerate(records):
            spectrum = reconstruct_spectrum(
                row, frequencies, row.get("spectrum_defaults") or {}
            )
            half_spectrum = reconstruct_spectrum(
                {**row, "beta_over_H": row["beta_over_H_half_step"]},
                frequencies,
                row.get("spectrum_defaults") or {},
            )
            spectral_rms = float(
                np.sqrt(
                    np.mean(
                        np.log10(
                            spectrum["sw"][support] / reference_spectrum["sw"][support]
                        )
                        ** 2
                    )
                )
            )
            peak_distance = float(
                np.linalg.norm(features([row])[0] - features([reference])[0])
            )
            exported.append(
                {
                    **row,
                    "group": group,
                    "benchmark": f"{group}{index + 1}",
                    "reference_point_id": reference["point_id"],
                    "peak_distance_dex": peak_distance,
                    "acoustic_spectrum_rms_dex": spectral_rms,
                    "delta_C_from_reference": row["C"] - reference["C"],
                    "delta_m6_GeV_from_reference": row["m6_GeV"] - reference["m6_GeV"],
                    "interpretation": "nearby_predicted_signals_not_statistical_indistinguishability",
                }
            )
            spectra_rows.extend(
                {
                    "group": group,
                    "benchmark": f"{group}{index + 1}",
                    "point_id": row["point_id"],
                    "f_Hz": float(f),
                    "omega_total_h2": float(total),
                    "omega_sw_h2": float(sound),
                    "omega_total_h2_beta_half_step": float(total_half),
                    "omega_sw_h2_beta_half_step": float(sound_half),
                }
                for f, total, sound, total_half, sound_half in zip(
                    frequencies,
                    spectrum["total"],
                    spectrum["sw"],
                    half_spectrum["total"],
                    half_spectrum["sw"],
                    strict=True,
                )
            )
    return exported, spectra_rows


def summarized_detector_intervals(cuts):
    """Une segmentos adjacentes no mesmo corte, preservando componentes e buracos."""
    groups = defaultdict(list)
    for row in cuts:
        groups[
            (
                row["detector"],
                row["m8_GeV"],
                row["m6_GeV"],
                row.get("beta_stencil", "h"),
            )
        ].append(row)
    result = []
    for (name, case, mass, stencil), rows in sorted(groups.items()):
        intervals = merge_intervals([(r["C_lower"], r["C_upper"]) for r in rows])
        for component, (low, high) in enumerate(intervals):
            result.append(
                {
                    "detector": name,
                    "beta_stencil": stencil,
                    "m8_GeV": case,
                    "m6_GeV": mass,
                    "component": component,
                    "C_lower": low,
                    "C_upper": high,
                    "interpretation": "local_sample_supported_interval_not_global_detectable_extrema",
                }
            )
    return result


def numerical_quality(dataset):
    """Conferências físicas independentes das escolhas de desenho."""
    accepted = [row for row in dataset.rows if row.get("nucleation_accepted")]
    beta = [row for row in dataset.rows if beta_valid(row)]
    residuals = [
        abs(row["S3_over_T"] - 140) for row in accepted if finite(row.get("S3_over_T"))
    ]
    phase_flags = []
    for row in accepted:
        jump, temperature = row.get("delta_phi_over_Tn"), row.get("Tn_GeV")
        if row.get("boundary_decision") == "yes" and finite(jump) and jump < 0.1:
            phase_flags.append(
                {
                    k: row.get(k)
                    for k in (
                        "point_id",
                        "m6_GeV",
                        "m8_GeV",
                        "C",
                        "Tn_GeV",
                        "phi_high_GeV",
                        "phi_low_GeV",
                        "delta_phi_over_Tn",
                        "alpha_trace",
                        "beta_over_H",
                    )
                }
            )
            phase_flags[-1]["diagnostic"] = (
                "small_field_jump_despite_phi_low_over_Tn_gt_1_check_phase_interpretation"
            )
            phase_flags[-1]["high_field_over_Tn"] = (
                abs(row["phi_high_GeV"]) / temperature
                if finite(row.get("phi_high_GeV")) and temperature
                else None
            )
    return {
        "accepted_nucleation_records": len(accepted),
        "maximum_absolute_S3_over_T_minus_140": max(residuals, default=None),
        "valid_beta_records": len(beta),
        "valid_beta_with_order_2": sum(row.get("beta_order") == 2 for row in beta),
        "valid_beta_with_half_step_saved": sum(
            finite(row.get("beta_over_H_half_step")) for row in beta
        ),
        "maximum_valid_beta_relative_change": max(
            (
                row["beta_relative_change"]
                for row in beta
                if finite(row.get("beta_relative_change"))
            ),
            default=None,
        ),
        "small_jump_selected_transitions": len(phase_flags),
        "phase_flags_modify_original_selection": False,
    }, phase_flags


def refinement_requests(dataset, profiles, *, threshold=10, limit=48):
    """Prioriza lacunas perto de amostras LISA positivas; não pede nova grade.

    Repetir falhas determinísticas não garante melhora. A tabela registra a
    necessidade e a razão; somente um comando explícito cria o plano físico.
    """
    requests = {}
    by_key = {coordinate(row): row for row in dataset.rows}
    for profile in profiles:
        if not profile.get("grid_C"):
            continue
        case, mass = float(profile["m8_GeV"]), float(profile["m6_GeV"])
        samples = sorted(profile["samples"], key=lambda item: item["C"])
        for index, sample in enumerate(samples):
            if sample.get("decision") != "unknown":
                continue
            neighbours = (
                samples[max(0, index - 1) : index] + samples[index + 1 : index + 2]
            )
            detected = [
                r
                for r in neighbours
                if finite(r.get("rho_sw")) and r["rho_sw"] >= threshold
            ]
            if not detected:
                continue
            key = case, mass, round(sample["C"], 12)
            row = by_key.get(key, {})
            if gw_valid(row):
                continue
            distance = min(abs(sample["C"] - item["C"]) for item in detected)
            requests[key] = {
                "m8_GeV": case,
                "m6_GeV": mass,
                "C": sample["C"],
                "point_id": sample["point_id"],
                "priority": 1,
                "reason": sample.get(
                    "reason", "GW_unresolved_next_to_detectable_sample"
                ),
                "quality_flags": row.get("quality_flags"),
                "distance_in_C": distance,
                "action": "diagnose_then_evaluate_in_separate_campaign",
            }
    return sorted(
        requests.values(),
        key=lambda row: (row["distance_in_C"], -row["m8_GeV"], row["m6_GeV"], row["C"]),
    )[:limit]
