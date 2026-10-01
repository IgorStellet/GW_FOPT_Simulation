#!/usr/bin/env bash
# Carregado pelos templates Slurm, depois de entrar na raiz do repositório.
# Define local_work. A pasta exclusiva permanece disponível para diagnóstico.

prepare_local_work() {
    local job_name="$1" candidate
    local -a candidates
    if [[ -n "${FOPT_SCRATCH_ROOT:-}" ]]; then
        # Uma escolha explícita deve funcionar ou produzir um erro claro.
        # Não se cria /scratch nem se substitui silenciosamente essa escolha.
        candidates=("$FOPT_SCRATCH_ROOT")
    else
        # Os caminhos disponibilizados pelo job têm preferência. /tmp permite
        # o piloto mesmo quando /scratch/local não existe no nó alocado.
        candidates=("${SLURM_TMPDIR:-}" "${TMPDIR:-}" /tmp)
    fi
    for candidate in "${candidates[@]}"; do
        if [[ -n "$candidate" && -d "$candidate" && -w "$candidate" && -x "$candidate" ]]; then
            if local_work="$(mktemp -d -- "$candidate/$job_name.XXXXXX")"; then
                printf 'Diretório temporário do job: %s\n' "$local_work"
                return 0
            fi
        fi
    done
    printf 'Não foi possível criar o diretório temporário. FOPT_SCRATCH_ROOT=%s\n' "${FOPT_SCRATCH_ROOT:-não definido}" >&2
    printf 'Escolha uma pasta local existente e gravável no nó, ou remova a escolha com unset FOPT_SCRATCH_ROOT para usar os temporários do job.\n' >&2
    return 1
}
