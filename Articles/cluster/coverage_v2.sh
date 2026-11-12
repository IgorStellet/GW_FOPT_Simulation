#!/usr/bin/env bash
# Execute na raiz do repositório. Cada comando preserva as campanhas v1.
set -euo pipefail
export FOPT_RESULTS_ROOT="${FOPT_RESULTS_ROOT:-$HOME/fopt-articles-v4}"
export FOPT_PYTHON="${FOPT_PYTHON:-$PWD/.venv-che311/bin/python}"
export FOPT_PYTHON_MODULE="${FOPT_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
if [[ ! -f Articles/cluster/coverage_observables.slurm ]]; then
    printf 'Entre na raiz GW_FOPT_Simulation antes de executar.\n' >&2; exit 2
fi
mkdir -p Articles/results/logs
plan="$FOPT_RESULTS_ROOT/coverage_plans/coverage_v2.json"
pilot="$FOPT_RESULTS_ROOT/coverage_plans/pilot_v2.json"
output="$FOPT_RESULTS_ROOT/coverage_observables_v2"
pilot_output="$FOPT_RESULTS_ROOT/coverage_pilot_v2"
command="${1:-help}"
case "$command" in
    plan|audit)
        inputs=(--boundaries "$FOPT_RESULTS_ROOT/boundaries_v1"
            --observables "$FOPT_RESULTS_ROOT/edge_observables_v1"
            --reach "$FOPT_RESULTS_ROOT/lisa_reach_v1")
        if [[ -d "$FOPT_RESULTS_ROOT/publication_refinement_v1" ]]; then
            inputs+=(--observables "$FOPT_RESULTS_ROOT/publication_refinement_v1")
        fi
        if [[ "$command" == plan ]]; then
            "$FOPT_PYTHON" -m Articles.figures.publication.coverage plan "${inputs[@]}" --output "$plan" --pilot-output "$pilot"
        else
            if [[ -d "$output" ]]; then inputs+=(--observables "$output"); fi
            "$FOPT_PYTHON" -m Articles.figures.publication.coverage audit "${inputs[@]}" --output "$FOPT_RESULTS_ROOT/coverage_audit_v2"
        fi
        ;;
    pilot)
        for task in 0 1 2; do
            "$FOPT_PYTHON" -m Articles.figures.publication.coverage run --plan "$pilot" --output "$pilot_output" --work-dir /tmp/fopt-dry-run-unused --task-index "$task" --shards-per-scenario 1 --dry-run
        done
        FOPT_COVERAGE_PLAN="$pilot" FOPT_COVERAGE_ROOT="$pilot_output" FOPT_COVERAGE_SHARDS=1 FOPT_COVERAGE_REUSE="$FOPT_RESULTS_ROOT/no-pilot-import" sbatch --export=ALL --array=0-2%3 --time=02:00:00 Articles/cluster/coverage_observables.slurm
        ;;
    submit)
        "$FOPT_PYTHON" -m Articles.figures.publication.coverage check-pilot --output "$pilot_output"
        for task in {0..11}; do
            "$FOPT_PYTHON" -m Articles.figures.publication.coverage run --plan "$plan" --output "$output" --work-dir /tmp/fopt-dry-run-unused --task-index "$task" --dry-run
        done
        FOPT_COVERAGE_PLAN="$plan" FOPT_COVERAGE_ROOT="$output" FOPT_COVERAGE_SHARDS=4 sbatch --export=ALL Articles/cluster/coverage_observables.slurm
        ;;
    status)
        for root in "$pilot_output" "$output"; do
            if [[ -d "$root" ]]; then "$FOPT_PYTHON" -m Articles.figures.publication.coverage status --output "$root"; fi
        done
        squeue -r -u "$USER"
        ;;
    figures)
        sbatch --export=ALL Articles/cluster/publication_v2.slurm
        ;;
    *)
        printf 'Uso: bash Articles/cluster/coverage_v2.sh plan|pilot|status|submit|audit|figures\n'
        ;;
esac
