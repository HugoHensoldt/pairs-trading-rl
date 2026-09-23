#!/bin/bash
# Modelos por combinação, estágio 1: resume as janelas do cache de amostras da LSTM em features
# pequenas (~45 colunas) por (pregão, combinação), junto com y e o lucro em pontos. Só CPU.
# Retomável: arquivos de features já existentes são pulados (o driver tenta até 3 vezes).
#
# Uso direto: sbatch slurm/submit_combo_features.sbatch   (o normal é slurm/submit_combo.sh)

#SBATCH --job-name=combo-feat
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

# cache de amostras e de P/L da LSTM (já existem no scratch); features novas em combo_feats
export LSTM_CACHE_DIR="${LSTM_CACHE_DIR:-/scratch/ppg-lncc/$USER/lstm_cache}"
export LSTM_PL_DIR="${LSTM_PL_DIR:-/scratch/ppg-lncc/$USER/lstm_pl_cache}"
export COMBO_FEAT_DIR="${COMBO_FEAT_DIR:-/scratch/ppg-lncc/$USER/combo_feats}"
# 54 combinações: sigma 1.4/1.6/1.8 (campanha v1) e 2.2/2.4/2.6 (v2) x Re 0.5/0.75/0.9 x Ri 0.5/1.0/1.5
export LSTM_SIGMA_GRID="${LSTM_SIGMA_GRID:-1.4,1.6,1.8,2.2,2.4,2.6}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""

cd "$SLURM_SUBMIT_DIR/src"
python combo_models.py --stage features --workers "${SLURM_CPUS_PER_TASK:-48}"
