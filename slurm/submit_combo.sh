#!/bin/bash
# Roda os modelos por combinação (gradient boosting + logística, 54 combinações, compra e venda
# separados) no Santos Dumont, encadeando jobs de CPU um por vez (MaxSubmit=1: sem --array e sem
# dois jobs na fila).
#
#   1) features : job na sequana_cpu_dev que resume o cache de amostras da LSTM em features
#                 (até 3 tentativas; é retomável).
#   2) train    : jobs de treino (20 tarefas: protocolos A e B x 5 x 2 lados) até não sobrar
#                 tarefa pendente; aborta se um job não concluir nenhuma tarefa nova.
#   3) aggregate: tabela-resumo em combo_runs/<tag>/combo_summary.csv.
#
# Uso (o driver precisa continuar vivo -> nohup/setsid):
#   nohup setsid bash slurm/submit_combo.sh > combo_campaign.log 2>&1 < /dev/null &
#   STAGES="train aggregate" bash slurm/submit_combo.sh      # retoma só o treino
#   COMBO_RUN_TAG=combo_v2 bash slurm/submit_combo.sh        # outra tag de resultados
#
# Pré-requisito: os caches lstm_cache e lstm_pl_cache do scratch com as 54 combinações
# (sigma 1.4/1.6/1.8 da campanha v1 e 2.2/2.4/2.6 da v2) para os pregões 1-488.
set -euo pipefail

STAGES="${STAGES:-features train aggregate}"
MAX_TRAIN_JOBS="${MAX_TRAIN_JOBS:-10}"

cd "$(dirname "$0")/.."

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-lstm/bin/activate"

export LSTM_CACHE_DIR="${LSTM_CACHE_DIR:-/scratch/ppg-lncc/$USER/lstm_cache}"
export LSTM_PL_DIR="${LSTM_PL_DIR:-/scratch/ppg-lncc/$USER/lstm_pl_cache}"
export COMBO_FEAT_DIR="${COMBO_FEAT_DIR:-/scratch/ppg-lncc/$USER/combo_feats}"
export COMBO_RUN_DIR="${COMBO_RUN_DIR:-/scratch/ppg-lncc/$USER/combo_runs}"
export COMBO_RUN_TAG="${COMBO_RUN_TAG:-combo_v1}"
export LSTM_SIGMA_GRID="${LSTM_SIGMA_GRID:-1.4,1.6,1.8,2.2,2.4,2.6}"
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p "/scratch/ppg-lncc/$USER/lstm-logs"

RUN_ROOT="$COMBO_RUN_DIR/$COMBO_RUN_TAG"

wait_for_empty_queue() {   # qualquer job meu na fila bloqueia (MaxSubmit=1)
    while [ -n "$(squeue --me --noheader --format='%j' 2>/dev/null)" ]; do
        echo "Aguardando fila liberar (ainda há job ativo: $(squeue --me --noheader --format='%j' | tr '\n' ' '))..."
        sleep 5
    done
}

n_done() {   # nº de tarefas concluídas (metrics.json existe)
    if [ -d "$RUN_ROOT" ]; then find "$RUN_ROOT" -name metrics.json 2>/dev/null | wc -l; else echo 0; fi
}

pending_tasks() {
    (cd src && OMP_NUM_THREADS=1 python combo_models.py --stage pending)
}

if [[ " $STAGES " == *" features "* ]]; then
    ok=0
    for attempt in 1 2 3; do
        wait_for_empty_queue
        echo "=== features: tentativa $attempt ($(date)) ==="
        if sbatch --wait slurm/submit_combo_features.sbatch; then ok=1; break; fi
        echo "AVISO: features terminou com erro/timeout (tentativa $attempt); é retomável."
    done
    if [ "$ok" -ne 1 ]; then
        echo "ERRO: features não concluiu em 3 tentativas. Veja /scratch/ppg-lncc/$USER/lstm-logs/combo-feat-*.err" >&2
        exit 1
    fi
fi

if [[ " $STAGES " == *" train "* ]]; then
    for job in $(seq 1 "$MAX_TRAIN_JOBS"); do
        pend="$(pending_tasks)"
        if [ -z "$pend" ]; then
            echo "=== Todas as tarefas concluídas. ==="
            break
        fi
        echo "=== Pendentes ($(echo "$pend" | wc -l)): $(echo "$pend" | tr '\n' ' ') ==="
        before="$(n_done)"

        wait_for_empty_queue
        echo "=== train: job $job ($(date)) ==="
        sbatch --wait slurm/submit_combo_train.sbatch || echo "AVISO: job de treino $job terminou com erro/timeout"

        if [ "$(n_done)" -le "$before" ]; then
            echo "ERRO: o job $job não concluiu nenhuma tarefa nova." >&2
            echo "Veja /scratch/ppg-lncc/$USER/lstm-logs/combo-train-*.err e retome com STAGES=\"train aggregate\"." >&2
            exit 1
        fi
    done
    if [ -n "$(pending_tasks)" ]; then
        echo "AVISO: MAX_TRAIN_JOBS=$MAX_TRAIN_JOBS atingido com tarefas pendentes; rode de novo com STAGES=\"train aggregate\"."
    fi
fi

if [[ " $STAGES " == *" aggregate "* ]]; then
    (cd src && OMP_NUM_THREADS=1 python combo_models.py --stage aggregate)
fi
