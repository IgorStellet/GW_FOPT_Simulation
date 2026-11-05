"""Legendas, tabelas e relatório que acompanham as figuras do artigo."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from Articles.collect_data import json_safe
from Articles.figures.__main__ import write_csv
from Articles.figures.publication.analysis import (
    numerical_quality,
    summarized_detector_intervals,
)


def write_products(
    dataset,
    profiles,
    windows,
    benchmarks,
    spectra,
    cuts,
    requests,
    availability,
    mesh_audit,
    writer,
    args,
):
    output = args.output
    quality, phase_flags = numerical_quality(dataset)
    tables = {
        "parameter_windows.csv": windows,
        "observables_complete.csv": dataset.rows,
        "benchmark_records.csv": benchmarks,
        "matched_spectrum_samples.csv": spectra,
        "detector_intervals.csv": cuts,
        "refinement_requests.csv": requests,
        "availability.csv": availability,
        "interpolation_audit.csv": mesh_audit,
        "transition_topology_flags.csv": phase_flags,
        "Table02_detector_windows.csv": summarized_detector_intervals(cuts),
        "detector_step_sensitivity.csv": [
            {
                key: row.get(key)
                for key in (
                    "point_id",
                    "m6_GeV",
                    "m8_GeV",
                    "C",
                    "beta_step_GeV",
                    "beta_over_H",
                    "beta_over_H_half_step",
                    "beta_relative_change",
                    "LISA_rho_sw",
                    "LISA_rho_sw_half_step",
                )
            }
            for row in dataset.rows
            if row.get("LISA_step_classification_agrees") is False
        ],
    }
    table_fields = (
        "benchmark",
        "m6_GeV",
        "m8_GeV",
        "C",
        "Tn_GeV",
        "alpha_trace",
        "beta_over_H",
        "LISA_rho_sw",
        "LISA_rho_sw_half_step",
        "beta_relative_change",
        "f_sw_peak_Hz",
        "omega_sw_peak_h2",
        "peak_distance_dex",
        "acoustic_spectrum_rms_dex",
        "point_id",
    )
    tables["Table01_benchmarks.csv"] = [
        {k: row.get(k) for k in table_fields} for row in benchmarks
    ]
    for name, records in tables.items():
        write_csv(output / name, records)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_audit": dataset.audit,
        "complete_table": dataset.audit.get("complete_point_table_received", False),
        "coverage": availability,
        "numerical_quality": quality,
        "interpolation": mesh_audit,
        "figure_files": [path.name for path in writer.files],
        "captions": writer.captions,
        "refinement_requests": len(requests),
        "refinement_execution": "not_submitted_by_the_renderer",
        "original_profiles": len(profiles),
        "planned_LISA_mass_cuts": sum(bool(p.get("grid_C")) for p in profiles),
        "assumptions": {
            "Lambda_GeV": 1000,
            "S3_over_T_target": 140,
            "selected_window": "accepted_nucleation_and_abs_phi_low_Tn_over_Tn_gt_1",
            "observation_years": args.observation_years,
            "snr_threshold": args.snr_threshold,
            "channel": "sound_waves_only_PIS",
            "PIS_reference": "https://arxiv.org/abs/2002.04615",
            "maximum_mass_gap_GeV": args.maximum_mass_gap,
            "maximum_C_gap": args.maximum_C_gap,
            "envelope_interpolation": "PCHIP_between_contiguous_5GeV_columns",
            "boundary_guide": args.boundary_guide,
            "boundary_curvature_weight": args.boundary_smoothness,
            "envelope_is_proof_of_connected_nucleation": False,
            "maximum_matched_peak_distance_dex": args.maximum_peak_distance,
            "statistical_degeneracy_established": False,
            "benchmarks_require_detection_at_both_beta_steps": True,
            "half_step_comparison_is_statistical_confidence_interval": False,
            "DECIGO_BBO_boundaries_have_LISA_refinement_accuracy": False,
        },
        "analysis_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*writer.files, *(output / name for name in tables)]
        },
    }
    (output / "publication_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    captions = [
        "# Figure captions",
        "",
        "Figures use the stored model and thermal prescription; no new bounce calculations are performed.",
        "",
    ]
    latex = ["% Captions: include with \\input; graphics require the graphicx package."]
    for name, caption in writer.captions.items():
        captions += [f"## {name}", "", caption, ""]
        escaped = _latex_caption(caption, args)
        latex += [
            r"\begin{figure*}[t]",
            r"\centering",
            rf"\includegraphics[width=\textwidth,height=0.68\textheight,keepaspectratio]{{{name}.pdf}}",
            rf"\caption{{{escaped}}}",
            rf"\label{{fig:{name}}}",
            r"\end{figure*}",
            "",
        ]
    (output / "CAPTIONS_EN.md").write_text("\n".join(captions), encoding="utf-8")
    (output / "captions.tex").write_text("\n".join(latex), encoding="utf-8")
    _benchmark_latex(output, benchmarks, args.observation_years)
    _portuguese_captions(output, args)
    _portuguese_guide(output, dataset, availability, requests, benchmarks)
    _gallery(output, writer)
    with zipfile.ZipFile(
        output / "article_figures_complete.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in {
                ".png",
                ".pdf",
                ".svg",
                ".csv",
                ".json",
                ".md",
                ".tex",
                ".html",
            }:
                archive.write(path, arcname=path.name)
    return report


def _latex_caption(caption, args):
    """Typeseta símbolos sem escapar o conteúdo das expressões matemáticas."""
    symbols = {
        "Lambda=1000 GeV": r"$\Lambda=1000\,\mathrm{GeV}$",
        "S3(Tn)/Tn approximately 140": r"$S_3(T_n)/T_n\simeq140$",
        "|phi_low(Tn)|/Tn>1": r"$|\phi_{\mathrm{low}}(T_n)|/T_n>1$",
        "log10(alpha_trace)": r"$\log_{10}\alpha_{\mathrm{tr}}$",
        "log10(beta/Hn)": r"$\log_{10}(\beta/H_n)$",
        "beta/Hn": r"$\beta/H_n$",
        "alpha_trace": r"$\alpha_{\mathrm{tr}}$",
        "beta(h/2)": r"$\beta(h/2)$",
        "beta(h)": r"$\beta(h)$",
        "v(Tc)/Tc=1": r"$v(T_c)/T_c=1$",
        "v(Tc)/Tc": r"$v(T_c)/T_c$",
        "f_sw,peak": r"$f_{\mathrm{sw,peak}}$",
        "h2Omega_sw,peak": r"$h^2\Omega_{\mathrm{sw,peak}}$",
        "m6=infinity": r"$m_6=\infty$",
        "m6=2000 GeV": r"$m_6=2000\,\mathrm{GeV}$",
        "m6": r"$m_6$",
        "m8": r"$m_8$",
        r"\infty": r"$\infty$",
        "Tn": r"$T_n$",
        "alpha": r"$\alpha_{\mathrm{tr}}$",
        "beta": r"$\beta$",
        "log10": r"$\log_{10}$",
        f"SNR>={args.snr_threshold:g}": rf"$\rho_{{\mathrm{{sw}}}}\geq{args.snr_threshold:g}$",
        f"SNR={args.snr_threshold:g}": rf"$\rho_{{\mathrm{{sw}}}}={args.snr_threshold:g}$",
    }
    pattern = re.compile(
        "("
        + "|".join(re.escape(token) for token in sorted(symbols, key=len, reverse=True))
        + ")"
    )
    return "".join(
        symbols[part] if part in symbols else re.sub(r"([_%&#])", r"\\\1", part)
        for part in pattern.split(caption)
    )


def _benchmark_latex(output, rows, years):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Calculated benchmark points used in Fig.~\ref{{fig:Fig06_matched_GW_signals}}. The final two columns of the upper block give the LISA acoustic-channel SNR with $\beta(h)$ and $\beta(h/2)$, respectively, for {years:g} years. The lower block gives acoustic peak coordinates and matching distances relative to A1 or B1 in each group. Distances are in dex; the RMS difference is evaluated over the reference signal's band above one percent of its peak.}}",
        r"\label{tab:matched-benchmarks}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"ID & $m_6$ [GeV] & $m_8$ [GeV] & $C$ & $T_n$ [GeV] & $\alpha_{\mathrm{tr}}$ & $\beta/H_n$ & $\rho_{\mathrm{sw}}(h)$ & $\rho_{\mathrm{sw}}(h/2)$ \\",
        r"\hline",
    ]
    for row in rows:

        def mass(value):
            return r"$\infty$" if math.isinf(float(value)) else f"{float(value):.1f}"

        lines.append(
            " & ".join(
                (
                    row["benchmark"],
                    mass(row["m6_GeV"]),
                    mass(row["m8_GeV"]),
                    f"{row['C']:.5f}",
                    f"{row['Tn_GeV']:.2f}",
                    f"{row['alpha_trace']:.4f}",
                    f"{row['beta_over_H']:.2f}",
                    f"{row['LISA_rho_sw']:.1f}",
                    f"{row['LISA_rho_sw_half_step']:.1f}",
                )
            )
            + r" \\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        "}",
        r"\par\medskip",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"ID & $f_{\mathrm{sw,peak}}$ [Hz] & $h^2\Omega_{\mathrm{sw,peak}}$ & $d_{\mathrm{peak}}$ & $d_{\mathrm{RMS}}$ \\",
        r"\hline",
    ]
    for row in rows:

        def scientific(value):
            mantissa, exponent = f"{value:.3e}".split("e")
            return rf"${mantissa}\times10^{{{int(exponent)}}}$"

        lines.append(
            " & ".join(
                (
                    row["benchmark"],
                    scientific(row["f_sw_peak_Hz"]),
                    scientific(row["omega_sw_peak_h2"]),
                    f"{row['peak_distance_dex']:.4f}",
                    f"{row['acoustic_spectrum_rms_dex']:.4f}",
                )
            )
            + r" \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    (output / "Table01_benchmarks.tex").write_text("\n".join(lines), encoding="utf-8")


def _portuguese_captions(output, args):
    captions = [
        "# Legendas em português",
        "",
        "## Figura 1 — Compensação entre mecanismos",
        "",
        "Guias das janelas selecionadas para os três valores de m8, a Lambda=1000 GeV. Menor m6 corresponde a maior coeficiente do operador de dimensão seis. As curvas são regularizadas dentro dos limites certificados e das restrições unilaterais das amostras internas; os dados brutos permanecem nas tabelas. O preenchimento é uma guia interpolada e não demonstra nucleação em todo o interior. Os segmentos tracejados representam o controle isolado m6=infinito.",
        "",
        "## Figuras 2, 3 e 4 — Mapas de observáveis",
        "",
        "Temperatura de nucleação, alpha de traço, beta/Hn e razão do campo na temperatura crítica, respectivamente para m8=infinito, 840.9 e 668.7 GeV. As escalas de cores são compartilhadas entre cenários; alpha e beta usam escala logarítmica. O contorno vinho representa SNR acústico do LISA igual a 10 para quatro anos. Branco indica falta de interpolação local sustentada pelos dados e não exclusão física. O corte em Tn e a razão em Tc são quantidades diferentes.",
        "",
        "## Figura 5 — Alcance dos detectores e isolinhas",
        "",
        "Cada linha corresponde a um m8. As colunas mostram isolinhas rotuladas de log10(alpha) e log10(beta/Hn), sobrepostas às regiões de alcance acústico de LISA, DECIGO e BBO para quatro anos e SNR>=10. As regiões são sobrepostas, não exclusivas. Os limites de DECIGO/BBO têm a resolução das amostras disponíveis, sem herdar a tolerância do refinamento específico do LISA. As guias cinzas não certificam a conectividade de toda a janela.",
        "",
        "## Figura 6 — Sinais próximos com parâmetros diferentes",
        "",
        "Espectros totais dos grupos A e B, selecionados pela proximidade dos picos acústicos entre diferentes m8. O grupo A mantém m6 fixo e o B permite massas diferentes. Os painéis inferiores comparam os picos acústicos às curvas PIS na frequência de cada sinal. As coordenadas, observáveis e distâncias espectrais estão na Tabela 1. A proximidade de curvas não constitui demonstração de indistinguibilidade estatística.",
        "Nos mapas, o tracejado vinho usa beta(h/2), já salvo na campanha; a linha contínua usa beta(h). As bandas claras dos espectros mostram essa variação numérica, não intervalos de confiança. Os benchmarks passam o limiar do LISA nos dois passos.",
        "",
    ]
    captions = [
        line.replace("quatro anos", f"{args.observation_years:g} anos")
        .replace("igual a 10", f"igual a {args.snr_threshold:g}")
        .replace("SNR>=10", f"SNR>={args.snr_threshold:g}")
        for line in captions
    ]
    (output / "CAPTIONS_PT.md").write_text("\n".join(captions), encoding="utf-8")


def _portuguese_guide(output, dataset, coverage, requests, benchmarks):
    total_gw = sum(row["GW_valid"] for row in coverage)
    total_alpha = sum(row["alpha_valid"] for row in coverage)
    total_beta = sum(row["beta_valid"] for row in coverage)
    lines = [
        "# Figuras do artigo — leitura e limites",
        "",
        f"Foram lidos {len(dataset.rows)} pontos, com alpha positivo válido em {total_alpha}, beta válido em {total_beta} e GW resolvido em {total_gw}.",
        "",
        "## Como abrir",
        "",
        "Abra index.html para ver as seis figuras e suas legendas. Os arquivos PDF e SVG são as versões vetoriais para o artigo; PNG é a versão de revisão. Table01_benchmarks.tex e captions.tex podem ser incluídos no manuscrito com LaTeX.",
        "",
        "## As seis figuras",
        "",
        "1. Compensação entre mecanismos: três envelopes no plano C versus (1 TeV/m6)^2, sem marcadores, rachuras ou linhas verticais. Menor m6 significa maior coeficiente do operador de dimensão seis.",
        "2–4. Para cada m8, mapas de Tn, alpha de traço, beta/Hn e v(Tc)/Tc. O contorno vinho representa SNR acústico do LISA igual a 10 para quatro anos. As escalas de cores são compartilhadas entre os três cenários.",
        "5. Faixas dos detectores e isolinhas de log10(alpha) e log10(beta/Hn), na organização inspirada na figura de referência. As cores dos detectores se sobrepõem e não definem classes exclusivas.",
        "6. Dois trios de sinais próximos, um com m6 fixo e outro permitindo massas diferentes. Em cima estão os espectros totais; embaixo, os picos acústicos e o alcance PIS dos detectores. As coordenadas e a proximidade quantitativa estão nas tabelas.",
        "",
        "## O que a interpolação significa",
        "",
        "As bordas são guias regularizadas sob restrições: a inferior fica entre uma amostra externa negativa e a interna positiva; a superior obedece às desigualdades inversas. Sem amostra externa permanece somente a restrição unilateral. A curvatura é penalizada, mas nenhuma desigualdade é relaxada. Os pontos brutos, limites e valores ajustados estão em parameter_windows.csv. PCHIP liga as estimativas em massas contíguas sem extrapolar. A faixa preenchida não demonstra que todo o interior nucleia. Os mapas preservam avaliações fracassadas como lacunas; branco não significa exclusão física.",
        "",
        "A seleção original é |phi_low(Tn)|/Tn > 1, além da nucleação aceita perto de S3/T=140. A razão em Tc é um observável separado. Alpha e beta calculados nas bordas não são automaticamente extremos no interior.",
        "",
        "O alcance é do canal acústico sob os ajustes PIS históricos já usados na campanha. Não é uma análise instrumental de todos os canais. DECIGO e BBO usam as amostras existentes; seus limites não herdaram a precisão de 0.01 em C do refinamento específico do LISA.",
        "",
        "## Pendências numéricas",
        "",
        f"Foram listadas {len(requests)} avaliações prioritárias em refinement_requests.csv, próximas de amostras LISA detectáveis. O renderizador não executou nem submeteu esses cálculos. Falhas determinísticas exigem diagnóstico; simplesmente repetir pode não melhorar o resultado.",
        "",
        "As lacunas continuam visíveis nos mapas e registradas em interpolation_audit.csv. publication_report.json guarda contagens, procedência, hipóteses e hashes. observables_complete.csv preserva a tabela de pontos usada nesta edição.",
        "transition_topology_flags.csv também lista transições com salto de campo pequeno apesar de |phi_low(Tn)|/Tn>1. Elas não são automaticamente descartadas: precisam de interpretação da história de fases antes de serem chamadas de transições eletrofracas fortes.",
        "",
        "## Semelhança dos espectros",
        "",
        "A seleção usa distância em log10 da frequência e amplitude do pico. A tabela também informa a diferença RMS dos espectros acústicos em dex na banda em que a referência supera 1% do pico. Isso quantifica proximidade de sinais previstos, não demonstra indistinguibilidade estatística.",
        "A linha tracejada vinho nos mapas usa beta(h/2), que já estava salvo. As faixas claras nos espectros mostram a variação entre as duas avaliações. Esses diagnósticos não são intervalos de confiança. Os benchmarks passam o limiar do LISA nos dois passos; detector_step_sensitivity.csv lista os demais pontos cuja classificação muda, sem alterar o critério nominal da campanha.",
        "",
    ]
    if benchmarks:
        lines += [
            "| ID | m6 [GeV] | m8 [GeV] | C | SNR LISA | distância do pico [dex] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in benchmarks:
            lines.append(
                f"| {row['benchmark']} | {row['m6_GeV']:g} | {row['m8_GeV']:g} | {row['C']:.5f} | {row['LISA_rho_sw']:.2f} | {row['peak_distance_dex']:.5f} |"
            )
    (output / "LEIA_ME.md").write_text("\n".join(lines), encoding="utf-8")


def _gallery(output, writer):
    sections = []
    for name, caption in writer.captions.items():
        links = " · ".join(
            f'<a href="{name}.{ext}">{ext.upper()}</a>' for ext in writer.formats
        )
        picture = (
            f'<img src="{name}.png" alt="{html.escape(name)}">'
            if "png" in writer.formats
            else ""
        )
        sections.append(
            f"<section><h2>{html.escape(name)}</h2>{links}{picture}<p>{html.escape(caption)}</p></section>"
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>FOPT article figures</title>'
        "<style>body{max-width:1080px;margin:40px auto;padding:0 22px;font:17px/1.55 Georgia,serif;color:#20252b}"
        "section{margin:40px 0 64px}img{display:block;width:100%;height:auto;margin:18px 0}h2{font-size:21px}"
        "a{color:#175b83}p{max-width:1000px}</style><h1>FOPT — article figures</h1>"
        '<p><a href="LEIA_ME.md">Guia em português</a> · <a href="Table01_benchmarks.csv">Benchmarks</a> · '
        '<a href="publication_report.json">Data audit</a></p>'
        + "".join(sections)
        + "</html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
