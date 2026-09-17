#!/bin/bash
# Sweep de TARGET_N_PASSADAS -- testa a hipótese "menos passadas ajuda"
# (ver docs/relatorio_resultados_v2.md, seção 5: mais chunks/mais treino
# em target_n_passadas=20 correlacionou com overfitting severo).
#
# Roda só o FOLD 1 (o menor, 8 dias de treino/12 validação/12 teste com
# MAX_PREGOES=80 -- mesmos limites do fold 1 da v2) em cada ponto do
# sweep, pra caber em ~1-2h de Santos Dumont em vez de ~8h nos 3 folds.
# Reaproveita 100% da lógica de slurm/submit_all_folds.sh (planejamento
# de chunks, resume, wait_for_empty_queue, avaliação) -- só varia
# TARGET_N_PASSADAS/RUN_TAG por iteração; MAX_PREGOES/
# THROUGHPUT_STEPS_PER_SEC ficam nos defaults (mesmos de sempre).
#
# IMPORTANTE: demora até ~2h no total (5 rodadas sequenciais -- a conta
# ppg-lncc tem MaxSubmit=1, então não dá pra paralelizar). Rode dentro de
# tmux/screen ou com nohup, não direto num SSH que pode cair:
#   tmux new -s sweep
#   bash slurm/submit_sweep.sh 2>&1 | tee slurm_sweep.log
#
# Cada ponto do sweep é independente (RUN_TAG=p<N> namespacea checkpoint/
# best_model/log CSV) -- se precisar interromper no meio, os pontos já
# concluídos ficam utilizáveis.
#
# Uso:
#   bash slurm/submit_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_N_PASSADAS_VALUES=(1 3 5 10 20)

for v in "${TARGET_N_PASSADAS_VALUES[@]}"; do
    echo ""
    echo "############################################################"
    echo "### Sweep: TARGET_N_PASSADAS=${v} (RUN_TAG=p${v})  $(date)"
    echo "############################################################"
    RUN_TAG="p${v}" TARGET_N_PASSADAS="${v}.0" FOLDS="1" \
        bash slurm/submit_all_folds.sh
done

echo ""
echo "=== Sweep completo: TARGET_N_PASSADAS em ${TARGET_N_PASSADAS_VALUES[*]} ($(date)) ==="
