"""Mapas e espectros a partir do SQLite salvo, sem recalcular fases/bounces."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from Articles.core_imports import (
    bbo_sensitivity_s_pis,
    decigo_sensitivity_s_pis,
    gw_f_sw_peak,
    gw_omega_total_h2,
    lisa_sensitivity_s_pis,
)

COLORS = ("#e9edf2", "#e88a52", "#238b72", "#9d5bb5")
LABELS = ("Não calculado", "Sem nucleação encontrada", "Nucleação aceita", "Inconclusivo")
SPECTRUM_COLORS = ("#2866b4", "#239274", "#8a4dab", "#ac5635", "#555555")


def load_campaign(folder):
    folder = Path(folder)
    database = folder / "scan.sqlite"
    if not database.exists():
        raise ValueError(f"Banco não encontrado: {database}")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")  # Um mesmo snapshot para pontos, transições e manifesto.
        points = [json.loads(row) for (row,) in db.execute("SELECT record FROM points ORDER BY point_id")]
        transitions = [json.loads(row) for (row,) in db.execute(
            "SELECT record FROM transitions ORDER BY point_id, transition_index"
        )]
        saved = db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
    manifest = json.loads(saved[0]) if saved else {}
    # A grade legada não grava scan_decision. Deriva a mesma classificação
    # dos registros físicos, somente na cópia em memória usada pelo gráfico.
    from Articles.adaptive_scan import nucleation_decision

    transitions_by_point = {}
    for tr in transitions:
        transitions_by_point.setdefault(tr["point_id"], []).append(tr)
    for point in points:
        if "scan_decision" not in point:
            point["scan_decision"] = nucleation_decision(point, transitions_by_point.get(point["point_id"], []))
    axes_path, state_path = folder / "matrix_axes.json", folder / "adaptive_state.json"
    axes = json.loads(axes_path.read_text(encoding="utf-8")) if axes_path.exists() else {
        "m6_GeV": sorted({float(p["m6_GeV"]) for p in points}, reverse=True),
        "C": sorted({p["C"] for p in points}),
        "m8_GeV": list(dict.fromkeys(p["m8_GeV"] for p in points)),
    }
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    return {"folder": folder, "points": points, "transitions": transitions,
            "manifest": manifest, "axes": axes, "state": state}


def select_transitions(transitions, max_spectra=3):
    """Escolhe Cs distintos e registra por que outros registros não são usados."""
    if max_spectra < 1:
        raise ValueError("max-spectra deve ser positivo.")
    valid, excluded = [], []
    for tr in transitions:
        reason = None
        if not all(k in tr for k in ("C", "m6_GeV", "m8_GeV", "point_id")):
            reason = "missing_point_coordinates"
        elif tr.get("trantype") != 1 or not tr.get("nucleation_accepted"):
            reason = "nucleation_not_accepted"
        elif tr.get("status") != "nucleated" or tr.get("quality_flags"):
            reason = tr.get("quality_flags") or "observables_unresolved"
        elif not all(isinstance(tr.get(k), (int, float)) and math.isfinite(tr[k]) and tr[k] > 0
                     for k in ("alpha_trace", "beta_over_H", "Tn_GeV", "g_star", "v_w")):
            reason = "missing_or_invalid_spectrum_inputs"
        if reason:
            excluded.append({"point_id": tr.get("point_id"), "C": tr.get("C"), "reason": reason})
        else:
            valid.append(tr)
    by_C = {}
    # Em um C compartilhado, prefere uma massa finita; seleção determinística.
    for tr in sorted(valid, key=lambda t: (t["C"], float(t["m6_GeV"]), t["point_id"])):
        by_C.setdefault(tr["C"], tr)
    cs = sorted(by_C)
    indices = sorted(set(np.linspace(0, len(cs) - 1, min(max_spectra, len(cs)), dtype=int))) if cs else []
    return [by_C[cs[i]] for i in indices], excluded


def reconstruct_spectrum(tr, f_Hz, defaults):
    """Conversão explícita Hz->mHz e eficiências registradas no manifesto."""
    return gw_omega_total_h2(
        1000.0 * np.asarray(f_Hz), alpha=tr["alpha_trace"], beta_over_H=tr["beta_over_H"],
        T_star=tr["Tn_GeV"], g_star=tr["g_star"], v_w=tr["v_w"],
        epsilon_turb=defaults.get("epsilon_turb", 0.05), kappa_coll=defaults.get("kappa_coll", 0.0),
    )


def _pyplot(folder):
    config = folder / ".matplotlib"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, folder, name):
    for extension in ("png", "svg"):
        temporary = folder / f"{name}.{extension}.tmp"
        fig.savefig(temporary, format=extension, dpi=170, facecolor="white")
        temporary.replace(folder / f"{name}.{extension}")


def _mass_label(value):
    return "inf" if math.isinf(float(value)) else f"{float(value):g}"


def _draw_map(ax, campaign, m8):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    masses = [float(m) for m in campaign["axes"]["m6_GeV"]]
    cs = np.asarray(campaign["axes"]["C"], dtype=float)
    points = [p for p in campaign["points"] if float(p["m8_GeV"]) == float(m8)]
    if not len(cs) or not masses:
        ax.text(0.5, 0.5, "Nenhum ponto salvo", ha="center", transform=ax.transAxes)
        return
    grid = np.zeros((len(cs), len(masses)), dtype=int)
    x_index, y_index = {m: i for i, m in enumerate(masses)}, {c: i for i, c in enumerate(cs)}
    for p in points:
        decision = p.get("scan_decision", "unknown")
        grid[y_index[p["C"]], x_index[float(p["m6_GeV"])]] = {"no": 1, "yes": 2}.get(decision, 3)
    delta = campaign["manifest"].get("grid", {}).get("C_range", [0, 10, 0.02])[2]
    edges = np.r_[cs[0] - delta / 2, (cs[:-1] + cs[1:]) / 2, cs[-1] + delta / 2]
    ax.pcolormesh(np.arange(len(masses) + 1) - 0.5, edges, grid,
                  cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(5) - 0.5, 4), shading="flat")
    if points:
        ax.set_ylim(min(p["C"] for p in points) - delta, max(p["C"] for p in points) + delta)
    indices = sorted({0, *np.linspace(0, len(masses) - 1, min(8, len(masses)), dtype=int)})
    ax.set_xticks(indices, [_mass_label(masses[i]) for i in indices], rotation=35 if len(indices) > 5 else 0)
    for section in campaign["state"].get("slices", []):
        if float(section["m8_GeV"]) != float(m8):
            continue
        for col in section.get("columns", []):
            x = x_index[float(col["m6_GeV"])]
            if "verified_seed_C" in col:
                ax.scatter(x, col["verified_seed_C"], facecolors="none", edgecolors="black", s=75, linewidths=1.4)
            for stop in col.get("C_stops", []):
                if stop["reason"] == "consecutive_no_nucleation":
                    ax.scatter([x] * len(stop["last_no_C"]), stop["last_no_C"], color="#333333", marker="x", s=22)
                elif stop["reason"] == "pilot_step_limit" and stop.get("first_after_walk_C") is not None:
                    offset = delta / 2 if stop["direction"] == "decreasing_C" else -delta / 2
                    ax.scatter(x, stop["first_after_walk_C"] + offset, color="#2866b4", marker="_", s=100)
    warnings = [p for p in points if p["status"] == "observables_unresolved" and p.get("scan_decision") == "yes"]
    if warnings:
        ax.scatter([x_index[float(p["m6_GeV"])] for p in warnings], [p["C"] for p in warnings],
                   facecolors="none", edgecolors="#ba2737", marker="D", s=55)
    ax.set_title(f"m8 = {_mass_label(m8)} GeV · {len(points)} pontos")
    ax.set_xlabel("m6 [GeV] — colunas discretas; inf desliga o operador")
    ax.set_ylabel("C")


def _map_figure(campaigns, folder, name):
    plt = _pyplot(folder)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    panels = [(campaign, m8) for campaign in campaigns for m8 in campaign["axes"]["m8_GeV"]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(max(7.8, 6 * len(panels)), 6.4), squeeze=False)
    for ax, (campaign, m8) in zip(axes[0], panels, strict=True):
        _draw_map(ax, campaign, m8)
    handles = [Patch(facecolor=c, label=s) for c, s in zip(COLORS, LABELS, strict=True)]
    handles += [Line2D([], [], marker="o", color="black", markerfacecolor="none", linestyle="none", label="Semente verificada"),
                Line2D([], [], marker="x", color="#333333", linestyle="none", label="Três ausências: parada"),
                Line2D([], [], marker="_", color="#2866b4", linestyle="none", label="Limite da janela do piloto"),
                Line2D([], [], marker="D", color="#ba2737", markerfacecolor="none", linestyle="none", label="Nucleou; observáveis sob revisão")]
    pilot = any(c["manifest"].get("grid", {}).get("column_limit") for c in campaigns)
    title = "Faixas de nucleação — amostragem parcial do piloto" if pilot else "Faixas de nucleação — pontos calculados"
    fig.suptitle(title, fontsize=15, y=0.98)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.012, "Janela ampliada dos C avaliados. Cinza indica ausência de cálculo; não é exclusão física.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.18, 1, 0.94))
    _save(fig, folder, name)
    plt.close(fig)


def plot_campaign(folder, max_spectra=3):
    campaign = load_campaign(folder)
    target = campaign["folder"] / "plots"
    target.mkdir(parents=True, exist_ok=True)
    plt = _pyplot(target)
    _map_figure([campaign], target, "nucleation_map")
    selected, excluded = select_transitions(campaign["transitions"], max_spectra)
    defaults = campaign["manifest"].get("spectrum_defaults", {})
    peaks = [gw_f_sw_peak(t["beta_over_H"], t["Tn_GeV"], t["g_star"], t["v_w"]) * 1e-3 for t in selected]
    f_Hz = np.geomspace(min(1e-5, min(peaks, default=1e-3) / 100),
                       max(10.0, max(peaks, default=1e-3) * 100), 800)
    detectors = (("LISA", lisa_sensitivity_s_pis, "#c03942"),
                 ("DECIGO", decigo_sensitivity_s_pis, "#d77b1e"),
                 ("BBO", bbo_sensitivity_s_pis, "#998300"))
    fig, (ax, comparison) = plt.subplots(1, 2, figsize=(14, 6))
    table, peak_amplitudes = [], []
    for i, tr in enumerate(selected):
        spectrum = reconstruct_spectrum(tr, f_Hz, defaults)
        color = SPECTRUM_COLORS[i % len(SPECTRUM_COLORS)]
        label = f"C={tr['C']:g}; m6={_mass_label(tr['m6_GeV'])} GeV"
        ax.loglog(f_Hz, spectrum["total"], color=color, lw=2, label=label)
        ax.loglog(f_Hz, spectrum["sw"], color=color, lw=1, linestyle=":", alpha=0.7)
        peak = gw_f_sw_peak(tr["beta_over_H"], tr["Tn_GeV"], tr["g_star"], tr["v_w"]) * 1e-3
        peak_spectrum = reconstruct_spectrum(tr, np.array([peak]), defaults)
        peak_amplitudes.append(float(peak_spectrum["sw"][0]))
        comparison.scatter(peak, peak_spectrum["sw"][0], color=color, edgecolors="white", s=75, zorder=5, label=label)
        for k, frequency in enumerate(f_Hz):
            table.append({"point_id": tr["point_id"], "C": tr["C"], "m6_GeV": tr["m6_GeV"],
                          "m8_GeV": tr["m8_GeV"], "f_Hz": frequency,
                          **{f"omega_{name}_h2": values[k] for name, values in spectrum.items()}})
    for name, sensitivity, color in detectors:
        comparison.loglog(f_Hz, sensitivity(1000 * f_Hz), color=color, lw=1.7, label=f"{name} — PIS acústica")
    if not selected:
        ax.text(0.5, 0.5, "Ainda não há espectros com observáveis resolvidos.\nConsulte os status e plots/selection.json.",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(fontsize=8.5, loc="best")
    comparison.legend(fontsize=8.2, loc="best")
    ax.set(title="Espectros selecionados: total e componente acústica", xlabel="f [Hz]", ylabel=r"$h^2\Omega_{\rm GW}(f)$")
    comparison.set(title="Picos acústicos e limiares dos detectores", xlabel=r"$f_{\rm sw,peak}$ [Hz]", ylabel=r"$h^2\Omega_{\rm sw,peak}$")
    comparison.set_ylim(min(1e-20, min(peak_amplitudes, default=1e-12) * 0.05),
                        max(1e-7, max(peak_amplitudes, default=0) * 10))
    for axis in (ax, comparison):
        axis.grid(True, which="major", alpha=0.2)
    state = campaign["state"].get("state", "sem estado")
    state_label = {"pilot_complete": "piloto concluído", "complete": "concluído",
                   "pilot_complete_with_unresolved": "piloto concluído com inconclusivos",
                   "complete_with_unresolved": "busca concluída com inconclusivos",
                   "paused_point_budget": "parcial — limite de pontos", "unresolved": "inconclusivo",
                   "interrupted": "interrompido", "running": "em cálculo"}.get(state, state)
    m8_label = ", ".join(_mass_label(m) for m in campaign["axes"]["m8_GeV"])
    fig.suptitle(f"Diagnóstico de GW · m8 = {m8_label} GeV · {state_label}", y=0.98)
    fig.text(0.5, 0.055, "Esquerda: total (linha contínua), acústico (pontilhado). Direita: compare os picos às curvas PIS.", ha="center", fontsize=9)
    fig.text(0.5, 0.025, "PIS é sensibilidade integrada de pico; não é ruído ponto a ponto nem SNR total. Frequências em Hz.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    _save(fig, target, "gw_spectra")
    plt.close(fig)
    temporary = target / "spectrum_samples.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        keys = ["point_id", "C", "m6_GeV", "m8_GeV", "f_Hz", "omega_sw_h2", "omega_turb_h2", "omega_coll_h2", "omega_total_h2"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table)
    temporary.replace(target / "spectrum_samples.csv")
    selection = {"selected": selected, "excluded": excluded, "frequency_unit": "Hz",
                 "selection_policy": "distinct_C_quantiles_prefer_finite_m6_on_ties",
                 "epsilon_turb": defaults.get("epsilon_turb", 0.05), "kappa_coll": defaults.get("kappa_coll", 0),
                 "matplotlib": importlib.metadata.version("matplotlib"),
                 "detector_interpretation": "acoustic_peak_vs_s_channel_PIS_not_total_SNR"}
    temporary = target / "selection.json.tmp"
    temporary.write_text(json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(target / "selection.json")
    print(f"Gráficos e amostras salvos em {target.resolve()}", flush=True)
    return selection


def plot_overview(root):
    root = Path(root)
    campaigns = [load_campaign(p.parent) for p in sorted(root.glob("m8-*/scan.sqlite"))]
    if campaigns:
        target = root / "plots"
        target.mkdir(parents=True, exist_ok=True)
        _map_figure(campaigns, target, "nucleation_overview")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Campanha ou pasta contendo m8-0/1/2.")
    parser.add_argument("--max-spectra", type=int, default=3)
    args = parser.parse_args(argv)
    if (args.output / "scan.sqlite").exists():
        plot_campaign(args.output, args.max_spectra)
    else:
        children = sorted(args.output.glob("m8-*/scan.sqlite"))
        if not children:
            parser.error("Nenhuma campanha encontrada na pasta informada.")
        for child in children:
            plot_campaign(child.parent, args.max_spectra)
        plot_overview(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
