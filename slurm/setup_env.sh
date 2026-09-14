#!/bin/bash
# Roda UMA VEZ no login node do Santos Dumont (não no compute node --
# compute node normalmente não tem acesso à internet pra baixar pacotes).
#
# Uso:
#   bash slurm/setup_env.sh
set -euo pipefail

# Confirmado via `module avail python` no Santos Dumont (set/2026). Sem
# isso, o venv cai no python3 default do sistema (3.6.8 -- velho demais:
# puxa stable-baselines3<2.0, que não suporta a API do gymnasium usada
# no pipeline).
module load python/3.10.16_sequana

ENV_DIR="$HOME/envs/pairs-rl"
rm -rf "$ENV_DIR"  # remove venv antigo (criado sem module load) se existir

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Ambiente criado em $ENV_DIR"
echo "O .sbatch já ativa esse mesmo caminho -- se você mudar ENV_DIR aqui,  ajuste lá também."
