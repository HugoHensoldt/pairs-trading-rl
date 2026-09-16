#!/bin/bash
# Submete os N_FOLDS folds em sequência, um job por vez -- necessário
# porque a conta ppg-lncc/QOS normal tem MaxSubmit=1 (só 1 job na fila
# por vez; um --array de N tasks conta como N jobs e é rejeitado).
#
# `sbatch --wait` bloqueia até o job daquele fold terminar (ou ser
# matado pelo --time) antes de submeter o próximo -- assim nunca há
# mais de 1 job nosso na fila ao mesmo tempo.
#
# Uso:
#   bash slurm/submit_all_folds.sh
set -euo pipefail

N_FOLDS=3  # mesmo N_FOLDS de src/rl_trading_pipeline.py -- ajuste junto se mudar lá

# Escolha de hoje (2026-09-16): usar bem menos que os 480 pregões
# disponíveis pra manter a rodada em ~3h48/12 chunks em vez de ~21,5h/68
# chunks com o dataset completo. Aumente quando quiser uma rodada maior
# -- exportado (não só setado) pra `--export=ALL` abaixo propagar pro
# sbatch, e usado tanto no cálculo do plano quanto no treino real.
export MAX_PREGOES="${MAX_PREGOES:-80}"

cd "$(dirname "$0")/.."

# Quantos jobs de 20min encadear POR FOLD -- não é mais um número fixo
# igual pra todos: cada fold precisa de um n_chunks_needed diferente
# pra chegar em TARGET_N_PASSADAS (default 20, ver
# generate_walk_forward_folds em rl_trading_pipeline.py), já que o
# número de dias de treino cresce a cada fold (janela expansiva). Um
# N_CHUNKS_PER_FOLD único pra todos os folds reintroduziria o mesmo
# desbalanceamento que causou overfitting no fold 1 (ver
# docs/relatorio_resultados.md): fold 1 (poucos dias) receberia chunks
# demais (overfit) e/ou o fold 3 (muitos dias) chunks de menos
# (undertraining). Por isso o plano é gerado pela MESMA função Python
# que dimensiona os folds, em vez de duplicar essa conta aqui em bash.
#
# Ajuste TARGET_N_PASSADAS/THROUGHPUT_STEPS_PER_SEC (env vars) se
# quiser recalibrar -- em especial THROUGHPUT_STEPS_PER_SEC depois de
# medir o fps real no node do Santos Dumont (ver fold 1 rodando antes
# de confiar no plano pros folds maiores).
echo "=== Calculando plano de chunks por fold (TARGET_N_PASSADAS=${TARGET_N_PASSADAS:-20.0}) ==="
declare -A CHUNKS_POR_FOLD
while read -r fold_num n_chunks; do
    CHUNKS_POR_FOLD["$fold_num"]="$n_chunks"
    echo "  Fold $fold_num: $n_chunks chunk(s)"
done < <(cd src && PRINT_FOLD_PLAN=1 python3 rl_trading_pipeline.py)

# Espera qualquer job pairs-rl-fold* nosso sair da fila antes de
# submeter o próximo -- cobre tanto sobra de execução anterior quanto
# um `sbatch` manual disparado por fora deste script (foi exatamente
# isso que causou AssocMaxSubmitJobLimit na primeira tentativa: um job
# de teste rodado manualmente ainda contava pro limite MaxSubmit=1).
wait_for_empty_queue() {
    while squeue --me --noheader --format="%j" 2>/dev/null | grep -q '^pairs-rl-fold'; do
        echo "Aguardando fila liberar (ainda há job pairs-rl-fold* ativo)..."
        sleep 5
    done
}

for i in $(seq 1 "$N_FOLDS"); do
    n_chunks="${CHUNKS_POR_FOLD[$i]:?fold $i não apareceu no plano do PRINT_FOLD_PLAN}"
    for chunk in $(seq 1 "$n_chunks"); do
        resume=1
        if [ "$chunk" -eq 1 ]; then resume=0; fi
        final=0
        if [ "$chunk" -eq "$n_chunks" ]; then final=1; fi

        wait_for_empty_queue
        echo "=== Submetendo fold $i/$N_FOLDS, chunk $chunk/$n_chunks ($(date)) ==="
        sbatch --wait \
            --job-name="pairs-rl-fold${i}" \
            --export=ALL,SLURM_ARRAY_TASK_ID="$i",RESUME_TRAINING="$resume",FINAL_CHUNK="$final" \
            slurm/submit_fold.sbatch
        echo "=== Fold $i, chunk $chunk concluído ($(date)) ==="
    done
done

echo "=== Todos os $N_FOLDS folds concluídos ==="
