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

cd "$(dirname "$0")/.."

for i in $(seq 1 "$N_FOLDS"); do
    echo "=== Submetendo fold $i/$N_FOLDS ($(date)) ==="
    SLURM_ARRAY_TASK_ID=$i sbatch --wait \
        --job-name="pairs-rl-fold${i}" \
        --export=ALL,SLURM_ARRAY_TASK_ID="$i" \
        slurm/submit_fold.sbatch
    echo "=== Fold $i concluído ($(date)) ==="
done

echo "=== Todos os $N_FOLDS folds concluídos ==="
