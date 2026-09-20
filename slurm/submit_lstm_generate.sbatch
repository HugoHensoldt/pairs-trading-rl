#!/bin/bash
# Estágio 1 da LSTM: gera (em paralelo, 1 processo por pregão) as amostras de
# todas as combinações (sigma, Re, Ri) do sweep e imprime o relatório de
# balanceamento de labels. Não usa GPU. Retomável: pregões/combinações já em
# cache (/scratch/.../lstm_cache) são pulados, então se estourar os 20 min
# basta rodar de novo (o submit_lstm.sh já tenta até 3 vezes).
#
# A grade pode ser trocada sem editar código, exportando antes do sbatch:
#   LSTM_SIGMA_GRID="0.5,1" LSTM_RE_GRID="0.5,0.75" LSTM_RI_GRID="0.5,1,2"
# (Re/Ri são FRAÇÕES do caminho até o alvo: 0.5 = 50%.)
#
# Uso direto: sbatch slurm/submit_lstm_generate.sbatch

#SBATCH --job-name=lstm-gen
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

export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export LSTM_CACHE_DIR="${LSTM_CACHE_DIR:-/scratch/ppg-lncc/$USER/lstm_cache}"
export LSTM_RUN_DIR="${LSTM_RUN_DIR:-/scratch/ppg-lncc/$USER/lstm_runs}"
export LSTM_RUN_TAG="${LSTM_RUN_TAG:-lstm_v1}"
export CUDA_VISIBLE_DEVICES=""   # este estágio não precisa (nem deve tocar) GPU
export TF_CPP_MIN_LOG_LEVEL=2

cd "$SLURM_SUBMIT_DIR/src"
python lstm_pipeline.py --stage report --workers "${SLURM_CPUS_PER_TASK:-48}"
