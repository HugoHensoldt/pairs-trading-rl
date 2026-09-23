#!/bin/bash
# Modelos por combinação, estágio 2: treina as tarefas pendentes (protocolo A = split atual,
# protocolo B = blocos móveis; compra e venda), 54 combinações x 2 modelos por tarefa, só CPU.
# Cada tarefa grava o resultado ao terminar (atômico) e as já concluídas são puladas; se o
# orçamento de tempo acabar, o driver submete outro job para o que sobrou.
#
# Uso direto: sbatch slurm/submit_combo_train.sbatch   (o normal é slurm/submit_combo.sh)

#SBATCH --job-name=combo-train
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --account=ppg-lncc
#SBATCH --output=/scratch/ppg-lncc/%u/lstm-logs/%x-%j.out
#SBATCH --error=/scratch/ppg-lncc/%u/lstm-logs/%x-%j.err

set -euo pipefail

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-lstm/bin/activate"

export LSTM_CACHE_DIR="${LSTM_CACHE_DIR:-/scratch/ppg-lncc/$USER/lstm_cache}"
export LSTM_PL_DIR="${LSTM_PL_DIR:-/scratch/ppg-lncc/$USER/lstm_pl_cache}"
export COMBO_FEAT_DIR="${COMBO_FEAT_DIR:-/scratch/ppg-lncc/$USER/combo_feats}"
export COMBO_RUN_DIR="${COMBO_RUN_DIR:-/scratch/ppg-lncc/$USER/combo_runs}"
export COMBO_RUN_TAG="${COMBO_RUN_TAG:-combo_v1}"
export LSTM_SIGMA_GRID="${LSTM_SIGMA_GRID:-1.4,1.6,1.8,2.2,2.4,2.6}"
# orçamento de tempo do processo (o job tem 1200 s): não começa tarefa nova depois de 80% dele
export COMBO_MAX_SECONDS="${COMBO_MAX_SECONDS:-1080}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1   # 1 thread por processo; o paralelismo é entre combinações

cd "$SLURM_SUBMIT_DIR/src"
python combo_models.py --stage train --workers "${SLURM_CPUS_PER_TASK:-48}"
