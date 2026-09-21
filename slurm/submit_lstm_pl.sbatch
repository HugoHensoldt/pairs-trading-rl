#!/bin/bash
# Estágio "pl" da LSTM: regera a heurística de todos os pregões e grava o LUCRO
# EM PONTOS de cada negociação (hoje o cache de amostras só guarda o sinal, y).
# É um cache LATERAL: não toca em lstm_cache nem nos treinos; cada arquivo
# pl_*.npz tem buy_pl/sell_pl na MESMA ordem de buy_y/sell_y das amostras, então
# alinha 1:1 com predictions.npz (ver load_sample_pl em src/lstm_pipeline.py).
# Se a amostra correspondente existe em lstm_cache, o alinhamento é conferido e
# qualquer divergência vira erro.
#
# Só CPU (sequana_cpu_dev), sem relatório serial (o do estágio generate é o que
# estourava os 20 min). Retomável: pregões/combinações já em lstm_pl_cache são
# pulados. Para tentar até 3 vezes use slurm/submit_lstm_pl.sh.
#
# Grade: por padrão a do experimento grid_wide (120 combinações), que CONTÉM as
# 27 do base/net_big/seed2. Troque sem editar código:
#   LSTM_SIGMA_GRID="1.4,1.6,1.8" LSTM_RE_GRID="0.5,0.75,0.9" LSTM_RI_GRID="0.5,1,1.5" sbatch slurm/submit_lstm_pl.sbatch
#
# Uso direto: sbatch slurm/submit_lstm_pl.sbatch

#SBATCH --job-name=lstm-pl
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
export LSTM_PL_DIR="${LSTM_PL_DIR:-/scratch/ppg-lncc/$USER/lstm_pl_cache}"
export LSTM_SIGMA_GRID="${LSTM_SIGMA_GRID:-1.0,1.2,1.4,1.6,1.8,2.0}"
export LSTM_RE_GRID="${LSTM_RE_GRID:-0.5,0.6,0.75,0.9}"
export LSTM_RI_GRID="${LSTM_RI_GRID:-0.5,0.75,1.0,1.25,1.5}"
export CUDA_VISIBLE_DEVICES=""   # só CPU
export TF_CPP_MIN_LOG_LEVEL=2

cd "$SLURM_SUBMIT_DIR/src"
python lstm_pipeline.py --stage pl --workers "${SLURM_CPUS_PER_TASK:-48}"
