#!/bin/bash
# Roda a v4 para VÁRIAS seeds, uma seed inteira (todos os folds) por vez.
# Cada seed usa RUN_TAG=v4_s<seed> e SEED=<seed>, então os artefatos não
# colidem (src/runs/v4_s0/, src/runs/v4_s1/, ...). Reaproveita
# slurm/submit_all_folds.sh sem alterá-lo (MaxSubmit=1: um job por vez).
#
# Se uma seed abortar (ex.: chunk não registrado no state.json), registra a
# falha e segue para a próxima; no fim, lista as seeds que falharam. Para
# retomar uma seed que falhou, rode direto:
#   RUN_TAG=v4_s<seed> SEED=<seed> START_CHUNK=<k> FOLDS="<i> ..." \
#       bash slurm/submit_all_folds.sh
#
# Uso (as env vars do pipeline são repassadas: VEC_ENV, TORCH_THREADS,
# BATCH_SIZE, N_STEPS, TOTAL_TIMESTEPS, EVAL_EVERY_STEPS, N_VAL, N_TEST...):
#   nohup bash slurm/submit_matrix.sh > /scratch/ppg-lncc/$USER/v4_matrix.log 2>&1 &
#   SEEDS="0 1 2 3 4" bash slurm/submit_matrix.sh
set -uo pipefail

SEEDS="${SEEDS:-0 1 2}"
cd "$(dirname "$0")/.."

falhas=""
for seed in $SEEDS; do
    echo "########## SEED $seed (RUN_TAG=v4_s${seed}) — $(date) ##########"
    if RUN_TAG="v4_s${seed}" SEED="$seed" bash slurm/submit_all_folds.sh; then
        echo "########## SEED $seed concluída — $(date) ##########"
    else
        echo "########## SEED $seed FALHOU — $(date) ##########" >&2
        falhas="$falhas $seed"
    fi
done

if [ -n "$falhas" ]; then
    echo "Seeds com falha:$falhas" >&2
    exit 1
fi
echo "=== Todas as seeds concluídas: $SEEDS ==="
