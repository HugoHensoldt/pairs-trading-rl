#!/bin/bash
# Estágio 2 da LSTM: UM job na sequana_gpu_dev (1 nó, 4x V100). As tarefas
# (fold, lado) ainda pendentes são distribuídas em round-robin pelas 4 GPUs;
# cada GPU roda a sua fila EM SEQUÊNCIA (as 4 GPUs em paralelo). Cada tarefa
# treina com o tempo que sobra do job, grava checkpoint (modelo + otimizador +
# early stopping) e, se não terminar, o próximo job retoma de onde parou. Quem
# lista as pendentes é `lstm_pipeline.py --stage pending`, na ordem
# fold1-buy, fold1-sell, fold2-buy, ... (as em andamento vêm primeiro).
#
# Não use --array: a conta tem MaxSubmit=1. Para encadear os jobs até tudo
# acabar use slurm/submit_lstm.sh.
#
# Acompanhar:
#   squeue --me
#   tail -f /scratch/ppg-lncc/$USER/lstm-logs/task-fold<k>-<lado>-<jobid>.out

#SBATCH --job-name=lstm-train
#SBATCH --partition=sequana_gpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --account=ppg-lncc
#SBATCH --output=/scratch/ppg-lncc/%u/lstm-logs/%x-%j.out
#SBATCH --error=/scratch/ppg-lncc/%u/lstm-logs/%x-%j.err

set -euo pipefail

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-lstm/bin/activate"

export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export LSTM_CACHE_DIR="${LSTM_CACHE_DIR:-/scratch/ppg-lncc/$USER/lstm_cache}"
export LSTM_RUN_DIR="${LSTM_RUN_DIR:-/scratch/ppg-lncc/$USER/lstm_runs}"
export LSTM_RUN_TAG="${LSTM_RUN_TAG:-lstm_v1}"

# Tempo total do job (--time=00:20:00 = 1200 s) e folga. Cada tarefa recebe como
# LSTM_TRAIN_MAX_SECONDS o que sobra do job naquele instante (contado a partir
# do início do processo Python, incluindo carga/normalização dos dados); o
# pipeline ainda reserva 60 s p/ a avaliação final. Tarefa que começaria com
# menos de MIN_SECONDS sobrando fica para o próximo job.
JOB_START=$(date +%s)
JOB_SECONDS="${LSTM_JOB_SECONDS:-1200}"
MARGIN_SECONDS=90
MIN_SECONDS=300

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=12   # 12 CPUs por GPU (padrão do Slurm nesta partição)

N_GPUS="${LSTM_GPUS:-4}"
BATCH_SIZE="${LSTM_BATCH_SIZE:-256}"
LOG_DIR="/scratch/ppg-lncc/$USER/lstm-logs"
mkdir -p "$LOG_DIR"

echo "=== GPUs visíveis ==="
nvidia-smi -L || true
python -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))" || true

cd "$SLURM_SUBMIT_DIR/src"

mapfile -t TASKS < <(python lstm_pipeline.py --stage pending)
if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "Nenhuma tarefa pendente."
    exit 0
fi

# fila de cada GPU (round-robin sobre as pendentes)
declare -a QUEUE
for ((g = 0; g < N_GPUS; g++)); do QUEUE[$g]=""; done
i=0
for t in "${TASKS[@]}"; do
    g=$((i % N_GPUS))
    QUEUE[$g]+="${t// /:};"      # "fold side" -> "fold:side;"
    i=$((i + 1))
done

run_gpu_queue() {   # $1 = índice da GPU, $2 = fila "fold:lado;fold:lado;..."
    local gpu="$1" queue="$2" item fold side remaining
    IFS=';' read -ra ITEMS <<< "$queue"
    for item in "${ITEMS[@]}"; do
        [ -z "$item" ] && continue
        fold="${item%%:*}"; side="${item##*:}"
        remaining=$(( JOB_SECONDS - MARGIN_SECONDS - ($(date +%s) - JOB_START) ))
        if [ "$remaining" -lt "$MIN_SECONDS" ]; then
            echo "GPU $gpu: só restam ${remaining}s; fold $fold/$side (e as seguintes da fila) ficam para o próximo job."
            return 0
        fi
        echo "=== GPU $gpu <- fold $fold, lado $side, ${remaining}s disponíveis ($(date)) ==="
        if CUDA_VISIBLE_DEVICES=$gpu LSTM_TRAIN_MAX_SECONDS=$remaining \
            python lstm_pipeline.py --stage train --fold "$fold" --side "$side" \
                --batch-size "$BATCH_SIZE" \
                > "$LOG_DIR/task-fold${fold}-${side}-${SLURM_JOB_ID}.out" 2>&1; then
            echo "GPU $gpu: fold $fold/$side ok ($(date))"
        else
            echo "AVISO: GPU $gpu, fold $fold/$side terminou com erro (ver task-fold${fold}-${side}-${SLURM_JOB_ID}.out)"
        fi
    done
}

PIDS=()
for ((g = 0; g < N_GPUS; g++)); do
    [ -z "${QUEUE[$g]}" ] && continue
    run_gpu_queue "$g" "${QUEUE[$g]}" &
    PIDS+=("$!")
done
# Espera as 4 filas; falhas não derrubam o job (o submit_lstm.sh decide se houve
# progresso olhando os state.json).
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "=== job de treino terminou ($(date)) ==="
