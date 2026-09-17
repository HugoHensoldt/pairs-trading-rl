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

# RUN_TAG: identifica rodadas diferentes do MESMO fold sem colidir
# artefatos (checkpoint/best_model/log CSV) -- usado pelo sweep de
# TARGET_N_PASSADAS (ver slurm/submit_sweep.sh). Sem a env var, ""
# reproduz o comportamento de sempre (v1/v2). Entra no --job-name abaixo,
# e por consequência no nome do arquivo .out (%x-%j.out).
export RUN_TAG="${RUN_TAG:-}"

# FOLDS: quais folds treinar nesta chamada, em ordem (default: todos os
# N_FOLDS). O sweep de TARGET_N_PASSADAS usa FOLDS="1" pra rodar só o
# fold 1 (menor) em cada ponto do sweep, mantendo os MESMOS folds (mesma
# função generate_walk_forward_folds, mesmo MAX_PREGOES) mas sem pagar o
# custo de treinar os 3.
FOLDS="${FOLDS:-$(seq 1 "$N_FOLDS")}"

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
# O cálculo do plano roda aqui no login node (fora do sbatch), então
# precisa ativar o mesmo venv/módulo que submit_fold.sbatch usa --
# sem isso, "python3" cai no Python do sistema, sem pandas/gymnasium/etc.
module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"

echo "=== Calculando plano de chunks por fold (TARGET_N_PASSADAS=${TARGET_N_PASSADAS:-20.0}, RUN_TAG=${RUN_TAG:-<vazio>}, FOLDS=${FOLDS}) ==="
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

for i in $FOLDS; do
    n_chunks="${CHUNKS_POR_FOLD[$i]:?fold $i não apareceu no plano do PRINT_FOLD_PLAN}"
    for chunk in $(seq 1 "$n_chunks"); do
        resume=1
        if [ "$chunk" -eq 1 ]; then resume=0; fi
        final=0
        if [ "$chunk" -eq "$n_chunks" ]; then final=1; fi

        wait_for_empty_queue
        echo "=== Submetendo fold $i (RUN_TAG=${RUN_TAG:-<vazio>}), chunk $chunk/$n_chunks ($(date)) ==="
        sbatch --wait \
            --job-name="pairs-rl-fold${i}${RUN_TAG:+-$RUN_TAG}" \
            --export=ALL,SLURM_ARRAY_TASK_ID="$i",RESUME_TRAINING="$resume",FINAL_CHUNK="$final",CHUNK_INDEX="$chunk" \
            slurm/submit_fold.sbatch
        echo "=== Fold $i, chunk $chunk concluído ($(date)) ==="
    done
done

echo "=== Folds concluídos: $FOLDS ==="
