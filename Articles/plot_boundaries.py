"""Mapas distinguem pontos calculados, faixa inferida e fronteiras pendentes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Articles.plot_results import _mass_label, _pyplot, _save, load_campaign


def draw_boundary_map(ax, campaign):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    state = json.loads((campaign["folder"] / "boundary_state.json").read_text(encoding="utf-8"))
    columns = state["columns"]
    masses = [float(c["m6_GeV"]) for c in columns]
    indices = {m: i for i, m in enumerate(masses)}
    ax.set_facecolor("#f0f2f5")
    for i, column in enumerate(columns):
        lower, upper = column.get("lower"), column.get("upper")
        if lower and upper:
            low, high = lower["inside_C"], upper["inside_C"]
            ax.add_patch(Rectangle((i - 0.43, low), 0.86, high - low, color="#cce6df", zorder=1))
        for edge in (lower, upper):
            if not edge:
                continue
            outside = edge.get("outside_C")
            if outside is None and edge["state"] == "unresolved":
                outside = state["plan"]["C_bounds"][0 if edge is lower else 1]
            if outside is None:
                continue
            low, high = sorted((edge["inside_C"], outside))
            unresolved = edge["state"] == "unresolved"
            ax.add_patch(Rectangle((i - 0.43, low), 0.86, high - low, facecolor="#dbcbea" if unresolved else "#efd6ad",
                                   edgecolor="#82599a" if unresolved else "#b58645",
                                   hatch="///" if unresolved else None, alpha=0.75, zorder=2))
    styles = {"inside": ("#18755b", "o"), "weak": ("#d99a23", "s"),
              "no": ("#c35a43", "x"), "unknown": ("#8253a2", "D")}
    plotted_cs = []
    for point in campaign["points"]:
        index = indices.get(float(point["m6_GeV"]))
        if index is None:
            continue
        if point.get("boundary_decision") == "unknown":
            kind = "unknown"
        elif point.get("selection_reason") == "weak_nucleation":
            kind = "weak"
        elif point.get("boundary_decision") == "yes":
            kind = "inside"
        else:
            kind = "no"
        color, marker = styles[kind]
        ax.scatter(index, point["C"], color=color, marker=marker, s=14, alpha=0.85, zorder=3)
        plotted_cs.append(point["C"])
    ticks = sorted({0, *np.linspace(0, max(0, len(masses) - 1), min(8, len(masses)), dtype=int)}) if masses else []
    ax.set_xticks(ticks, [_mass_label(masses[i]) for i in ticks], rotation=35, ha="right")
    ax.set_xlim(-0.6, max(0.6, len(masses) - 0.4))
    if plotted_cs:
        lo, hi = state["plan"]["C_bounds"]
        ax.set_ylim(max(lo - 0.03, min(plotted_cs) - 0.1), min(hi + 0.03, max(plotted_cs) + 0.1))
    ax.set_xlabel(r"$m_6$ [GeV]; inf desliga o operador")
    ax.set_ylabel("C")
    criterion = state["plan"]["criterion"]
    label = f"v(Tn)/Tn > {state['plan']['strong_threshold']:g}" if criterion == "strong" else "nucleação aceita"
    m8 = float(state["plan"]["m8_GeV"])
    ax.set_title(f"m8 = {_mass_label(m8)} GeV · {label}\n{state['state']}", fontsize=10)
    ax.grid(alpha=0.18)
    return [
        Line2D([], [], color=styles[k][0], marker=styles[k][1], linestyle="none", label=label)
        for k, label in (("inside", "Dentro do critério"), ("weak", "Nucleação fraca"),
                         ("no", "Sem nucleação encontrada"), ("unknown", "Inconclusivo / timeout"))
    ] + [Patch(facecolor="#cce6df", label="Faixa inferida; interior não amostrado"),
         Patch(facecolor="#efd6ad", label="Bracket da fronteira"),
         Patch(facecolor="#dbcbea", hatch="///", label="Fronteira pendente")]


def _figure(campaigns, target, name):
    target.mkdir(parents=True, exist_ok=True)
    plt = _pyplot(target)
    fig, axes = plt.subplots(1, len(campaigns), figsize=(max(8, 5 * len(campaigns)), 6.7), squeeze=False)
    for ax, campaign in zip(axes[0], campaigns):
        handles = draw_boundary_map(ax, campaign)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Fronteiras da região selecionada — somente marcadores são pontos calculados", fontsize=11)
    fig.tight_layout(rect=(0, 0.19, 1, 0.94))
    _save(fig, target, name)
    plt.close(fig)


def plot_boundary_campaign(folder):
    folder = Path(folder)
    _figure([load_campaign(folder)], folder / "plots", "boundary_map")


def plot_boundary_overview(root):
    root = Path(root)
    campaigns = [load_campaign(p.parent) for p in sorted(root.glob("m8-*/boundary_state.json"))]
    if campaigns:
        _figure(campaigns, root / "plots", "boundary_overview")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.output / "boundary_state.json").exists():
        plot_boundary_campaign(args.output)
    else:
        folders = sorted(p.parent for p in args.output.glob("m8-*/boundary_state.json"))
        if not folders:
            parser.error("Nenhuma campanha de fronteiras encontrada.")
        for folder in folders:
            plot_boundary_campaign(folder)
        plot_boundary_overview(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
