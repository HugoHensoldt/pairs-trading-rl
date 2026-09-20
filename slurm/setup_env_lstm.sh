#!/bin/bash
# Cria o venv da LSTM (TensorFlow com GPU) no Santos Dumont. Roda UMA VEZ no
# login node (compute node não tem internet). Separado do venv do RL
# (pairs-rl, torch CPU-only) porque aqui precisamos do TensorFlow com as
# bibliotecas CUDA (pip "tensorflow[and-cuda]" traz CUDA/cuDNN embutidos, então
# não depende de `module load cuda`; o driver 560.35 suporta CUDA 12.x).
#
# Uso:
#   bash slurm/setup_env_lstm.sh
set -euo pipefail

module load python/3.10.16_sequana

# /scratch e não /prj (mesma razão do setup_env.sh: lag de propagação no /prj)
ENV_DIR="/scratch/ppg-lncc/$USER/envs/pairs-lstm"
rm -rf "$ENV_DIR"
mkdir -p "/scratch/ppg-lncc/$USER/lstm-logs"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install "tensorflow[and-cuda]" pandas numpy scikit-learn

echo ""
echo "Ambiente criado em $ENV_DIR"
# No login node não há GPU: só confere que o TensorFlow importa. A conferência
# das GPUs acontece no início de cada job de treino (ver o .out do job).
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
