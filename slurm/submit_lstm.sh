#!/bin/bash
# Roda a LSTM inteira no Santos Dumont, encadeando jobs um por vez (a conta
# ppg-lncc tem MaxSubmit=1: sem --array e sem dois jobs na fila).
#
#   1) generate : job na sequana_cpu_dev que gera as amostras do sweep (cache)
#                 e imprime o relatório de balanceamento. Até 3 tentativas
#                 (é retomável: só processa o que falta).
#   2) train    : jobs na sequana_gpu_dev, cada um distribuindo as tarefas pendentes pelas 4 GPUs
#                 (fold, lado), uma fila por GPU, com checkpoint; repete
#                 até não sobrar tarefa pendente (5 folds x 2 lados = 10).
#                 Aborta se um job não avançar nenhum checkpoint.
#   3) aggregate: tabela final de métricas (val e teste, por fold).
#
# Uso:
#   bash slurm/submit_lstm.sh                       # tudo
#   STAGES="generate" bash slurm/submit_lstm.sh     # só gera + relatório (para fechar a grade)
#   STAGES="train aggregate" bash slurm/submit_lstm.sh   # retoma o treino
#   LSTM_RUN_TAG=lstm_v2 LSTM_RI_GRID="0.5,1,2" bash slurm/submit_lstm.sh
#
# Variáveis (todas opcionais): STAGES, MAX_TRAIN_JOBS (teto de segurança, 30),
# LSTM_RUN_TAG, LSTM_SIGMA_GRID / LSTM_RE_GRID / LSTM_RI_GRID, LSTM_N_SPLITS,
# LSTM_EXCLUDE_ORDERS, LSTM_GPUS, LSTM_BATCH_SIZE, LSTM_TRAIN_MAX_SECONDS.
set -euo pipefail

STAGES="${STAGES:-generate train aggregate}"
MAX_TRAIN_JOBS="${MAX_TRAIN_JOBS:-30}"

cd "$(dirname "$0")/.."

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-lstm/bin/activate"

export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export LSTM_CACHE_DIR="${LSTM_CACHE_DIR:-/scratch/ppg-lncc/$USER/lstm_cache}"
export LSTM_RUN_DIR="${LSTM_RUN_DIR:-/scratch/ppg-lncc/$USER/lstm_runs}"
export LSTM_RUN_TAG="${LSTM_RUN_TAG:-lstm_v1}"
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "/scratch/ppg-lncc/$USER/lstm-logs"

RUN_ROOT="$LSTM_RUN_DIR/$LSTM_RUN_TAG"

wait_for_empty_queue() {
    while squeue --me --noheader --format="%j" 2>/dev/null | grep -q '^lstm-'; do
        echo "Aguardando fila liberar (ainda há job lstm-* ativo)..."
        sleep 5
    done
}

progress_signature() {   # muda sempre que algum checkpoint/estado avança
    if [ -d "$RUN_ROOT" ]; then
        find "$RUN_ROOT" -name state.json -exec cat {} + 2>/dev/null | cksum
    else
        echo "vazio"
    fi
}

# pending/aggregate rodam no login node (sem GPU). CUDA_VISIBLE_DEVICES vai só no
# prefixo do comando: se fosse exportado, o sbatch o herdaria e esconderia as GPUs.
pending_tasks() {
    (cd src && CUDA_VISIBLE_DEVICES="" python lstm_pipeline.py --stage pending)
}

if [[ " $STAGES " == *" generate "* ]]; then
    ok=0
    for attempt in 1 2 3; do
        wait_for_empty_queue
        echo "=== generate: tentativa $attempt ($(date)) ==="
        if sbatch --wait slurm/submit_lstm_generate.sbatch; then ok=1; break; fi
        echo "AVISO: generate terminou com erro/timeout (tentativa $attempt); o cache é retomável."
    done
    if [ "$ok" -ne 1 ]; then
        echo "ERRO: geração não concluiu em 3 tentativas. Veja /scratch/ppg-lncc/$USER/lstm-logs/." >&2
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
        echo "=== Pendentes: $(echo "$pend" | tr '\n' ';') ==="
        before="$(progress_signature)"

        wait_for_empty_queue
        echo "=== train: job $job ($(date)) ==="
        sbatch --wait slurm/submit_lstm_train.sbatch || echo "AVISO: job de treino $job terminou com erro/timeout"

        if [ "$(progress_signature)" = "$before" ]; then
            echo "ERRO: o job $job não avançou nenhum checkpoint." >&2
            echo "Causas comuns: uma época sozinha maior que o orçamento de tempo (aumente LSTM_BATCH_SIZE" >&2
            echo "ou reduza a grade), falta de memória, ou GPU/TensorFlow indisponível. Veja os" >&2
            echo "task-fold*-*.out em /scratch/ppg-lncc/$USER/lstm-logs/ e retome com STAGES=\"train aggregate\"." >&2
            exit 1
        fi
    done
    if [ -n "$(pending_tasks)" ]; then
        echo "AVISO: MAX_TRAIN_JOBS=$MAX_TRAIN_JOBS atingido com tarefas pendentes; rode de novo com STAGES=\"train aggregate\"."
    fi
fi

if [[ " $STAGES " == *" aggregate "* ]]; then
    (cd src && CUDA_VISIBLE_DEVICES="" python lstm_pipeline.py --stage aggregate)
fi
