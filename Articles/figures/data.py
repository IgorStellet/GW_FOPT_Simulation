"""Leitura dos resultados e tabela física comum a todas as figuras.

Não há cálculo de fases, ações ou derivadas aqui. O SQLite é lido pela rotina
já existente; v(Tc) só é recuperado de dados efetivamente armazenados.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Articles.boundary_physics import classify
from Articles.boundary_scan import readonly_source_snapshot, reuse_compatible
from Articles.collect_data import Settings, provenance
from Articles.plot_results import (
    load_campaign,
    reconstruct_spectrum,
    select_transitions,
)

DETECTOR_NAMES = ("LISA", "DECIGO", "BBO")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def coordinate(row):
    return (float(row["m8_GeV"]), float(row["m6_GeV"]), round(float(row["C"]), 12))


def gw_valid(transition):
    """Reutiliza a seleção física do plotter anterior, sem mudar suas regras."""
    selected, _ = select_transitions([transition], 1)
    return bool(selected)


def critical_ratio(transition, details):
    """Usa o VEV crítico salvo; nunca substitui Tc por Tn ou extrapola fases."""
    temperature = transition.get("Tc_GeV")
    if not finite(temperature) or temperature <= 0:
        return None, "missing_Tc"
    field = transition.get("phi_low_Tc_GeV")
    if finite(field):
        return abs(field) / temperature, "transition_record"
    matches = []
    for item in details.get("critical_transitions", []):
        if not finite(item.get("Tcrit")) or not math.isclose(item["Tcrit"], temperature, rel_tol=1e-7, abs_tol=1e-5):
            continue
        if any(str(item.get(side + "_phase")) != str(transition[side + "_phase"])
               for side in ("high", "low") if side + "_phase" in transition):
            continue
        vev = np.asarray(item.get("low_vev", []), dtype=float).ravel()
        if vev.size == 1 and np.isfinite(vev[0]):
            matches.append(abs(float(vev[0])) / temperature)
    if len(matches) == 1:
        return matches[0], "critical_transitions"
    return None, "ambiguous_critical_transition" if matches else "critical_VEV_not_saved"


def details_for(folder, point):
    relative = point.get("details_path")
    if not relative:
        return {}
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError(f"details_path fora da campanha: {relative}")
    if not path.is_file():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def detector_metrics(transition, defaults, years=4.0):
    """SNR do CANAL ACÚSTICO sob os ajustes PIS de Schmitz (1 ano).

    rho_s = sqrt(t_obs/yr) Omega_s,peak / Omega_PIS^s. Não é SNR total:
    colisões, turbulência e canais cruzados não entram nesta classificação.
    As sensibilidades vêm do núcleo, com a conversão explícita Hz -> mHz.
    """
    if not gw_valid(transition):
        return {}
    from Articles.core_imports import (
        bbo_sensitivity_s_pis,
        decigo_sensitivity_s_pis,
        gw_f_sw_peak,
        lisa_sensitivity_s_pis,
    )

    peak_mHz = gw_f_sw_peak(transition["beta_over_H"], transition["Tn_GeV"], transition["g_star"], transition["v_w"])
    peak = float(reconstruct_spectrum(transition, np.array([peak_mHz * 1e-3]), defaults)["sw"][0])
    result = {"f_sw_peak_Hz": float(peak_mHz * 1e-3), "omega_sw_peak_h2": peak}
    for name, sensitivity in zip(DETECTOR_NAMES, (lisa_sensitivity_s_pis, decigo_sensitivity_s_pis, bbo_sensitivity_s_pis), strict=True):
        pis = float(sensitivity(np.array([peak_mHz]))[0])
        result[f"{name}_rho_sw"] = math.sqrt(years) * peak / pis
    return result


@dataclass
class Dataset:
    rows: list
    transitions: list
    columns: dict
    campaigns: list
    audit: dict

    @property
    def cases(self):
        return sorted({float(row["m8_GeV"]) for row in self.rows}, reverse=True)


def campaign_folders(root):
    root = Path(root).resolve()
    if (root / "scan.sqlite").is_file():
        return [root]
    result = sorted(path.parent for path in root.glob("m8-*/scan.sqlite"))
    if not result:
        raise ValueError(f"Nenhum scan.sqlite encontrado em {root}")
    return result


def load_dataset(boundaries, observables=(), *, years=4.0, read_details=True, allow_running=False, allow_unverified=False):
    campaigns, columns, state_log, candidates = [], {}, [], defaultdict(list)
    seen = set()
    current = provenance(Settings(), {})
    gw_source = "src/CosmoTransitions/gravitational_Waves.py"
    unverified = []
    for root in (boundaries, *observables):
        for folder in campaign_folders(root):
            if folder in seen:
                continue
            seen.add(folder)
            # Reutiliza o snapshot seguro do v4: mode=ro sozinho ainda pode
            # criar WAL/SHM numa origem WAL. O plotter abre só a cópia local.
            with (tempfile.TemporaryDirectory(prefix="figure-read-") as scratch,
                  readonly_source_snapshot(folder, Path(scratch)) as database):
                snapshot = Path(database.execute("PRAGMA database_list").fetchone()[2]).parent
                campaign = load_campaign(snapshot)
            campaign["folder"] = folder
            manifest = campaign["manifest"]
            if not manifest.get("source_sha256"):
                if not allow_unverified:
                    raise ValueError(f"Campanha sem hashes para auditoria: {folder}. --allow-unverified produz somente uma prévia.")
                unverified.append(str(folder))
            else:
                reuse_compatible(manifest, current, "boundaries")
                if manifest["source_sha256"].get(gw_source) != current["source_sha256"].get(gw_source):
                    raise ValueError(f"Rotinas GW diferentes da campanha: {folder}")
            state_path = folder / "boundary_state.json"
            if not state_path.exists():
                state_path = folder / "interior_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            if state.get("state") in ("running", "initializing") and not allow_running:
                raise ValueError(f"Campanha em execução: {folder}. Use --allow-running-preview somente para diagnóstico.")
            state_log.append({"folder": str(folder), "state": state.get("state", "not_recorded"),
                              "fingerprint": manifest.get("fingerprint"), "git_commit": manifest.get("git_commit")})
            if state_path.name == "boundary_state.json":
                case = float(state["plan"]["m8_GeV"])
                columns[case] = state["columns"]
            by_point = defaultdict(list)
            for transition in campaign["transitions"]:
                by_point[transition["point_id"]].append(transition)
            for point in campaign["points"]:
                trs = by_point[point["point_id"]]
                decision, reason = classify(point, trs, "strong", 1.0)
                accepted = [tr for tr in trs if tr.get("nucleation_accepted") and tr.get("trantype") == 1]
                primary = min(accepted, key=lambda tr: tr.get("transition_index", 0), default={})
                rank = (int(gw_valid(primary)) if primary else 0, int(decision == "yes"), int(decision != "unknown"))
                candidates[coordinate(point)].append((rank, point, primary, campaign, decision, reason, trs))
            campaigns.append(campaign)
    rows, all_transitions, conflicts = [], [], []
    for key, records in sorted(candidates.items(), key=lambda item: tuple(-x for x in item[0])):
        # Uma falha nova nunca apaga uma observação válida anterior.
        chosen = max(records, key=lambda record: record[0])
        _, point, primary, campaign, decision, reason, trs = chosen
        disagree = {record[4] for record in records} >= {"yes", "no"}
        if disagree:
            conflicts.append({"m8_GeV": key[0], "m6_GeV": key[1], "C": key[2]})
        row = {**point, **primary, "point_status": point["status"], "source_folder": str(campaign["folder"]),
               "source_fingerprint": campaign["manifest"].get("fingerprint"),
               "boundary_decision": "unknown" if disagree else decision,
               "selection_reason": "conflicting_sources" if disagree else reason,
               "source_conflict": disagree, "primary_transition_policy": "first_accepted_in_thermal_history"}
        row["gw_resolved"] = bool(primary and gw_valid(primary) and not disagree)
        if primary:
            Tn, low = primary.get("Tn_GeV"), primary.get("phi_low_GeV")
            if finite(Tn) and Tn > 0 and finite(low):
                row["vev_over_Tn"] = abs(low) / Tn
            details = details_for(campaign["folder"], point) if read_details else {}
            row["vev_over_Tc"], row["critical_ratio_source"] = critical_ratio(primary, details)
            Tc = primary.get("Tc_GeV")
            if finite(Tc) and finite(Tn) and Tc >= Tn > 0:
                row["supercooling_fraction"] = 1 - Tn / Tc
            if row["gw_resolved"]:
                row.update(detector_metrics(primary, campaign["manifest"].get("spectrum_defaults", {}), years))
        if finite(point.get("V_at_v_GeV4")) and finite(point.get("V_at_origin_GeV4")):
            row["delta_V0_GeV4"] = point["V_at_v_GeV4"] - point["V_at_origin_GeV4"]
        row["spectrum_defaults"] = campaign["manifest"].get("spectrum_defaults", {})
        row["eft6_TeV_inverse_squared"] = (1000 / float(point["m6_GeV"])) ** 2
        rows.append(row)
        all_transitions.extend({**tr, "source_folder": str(campaign["folder"])} for tr in trs)
    controls = {coordinate(row): row for row in rows}
    for row in rows:
        case, mass, coupling = coordinate(row)
        polynomial = controls.get((case, mass, 0.0))
        measure = controls.get((math.inf, math.inf, coupling))
        row["cooperation_verified"] = (
            row["boundary_decision"] == "yes" and polynomial is not None and measure is not None
            and polynomial["boundary_decision"] == measure["boundary_decision"] == "no"
        )
        row["pure_controls_available"] = polynomial is not None and measure is not None
    audit = {"sources": state_log, "unverified_sources": unverified, "conflicting_coordinates": conflicts,
             "synthetic_data": any(campaign["manifest"].get("synthetic_data", False) for campaign in campaigns),
             "points": len(rows), "resolved_GW_points": sum(row["gw_resolved"] for row in rows),
             "critical_ratios_available": sum(finite(row.get("vev_over_Tc")) for row in rows),
             "statuses": dict(Counter(row["status"] for row in rows)),
             "analysis_source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in Path(__file__).parent.glob("*.py")},
             "assumptions": {"observation_years": years, "SNR_channel": "sound_waves_only_PIS",
                             "PIS_reference": "https://arxiv.org/abs/2002.04615", "PIS_reference_years": 1,
                             "strong_selection": "abs(phi_low(Tn))/Tn > 1", "frequency_unit": "Hz",
                             "mass_axis": "physical_GeV_or_explicit_inverse_squared_strength",
                             "uncomputed_region": "never_interpreted_as_excluded"}}
    return Dataset(rows, all_transitions, columns, campaigns, audit)
