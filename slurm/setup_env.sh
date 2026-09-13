#!/bin/bash
# Roda UMA VEZ no login node do Santos Dumont (não no compute node --
# compute node normalmente não tem acesso à internet pra baixar pacotes).
#
# Uso:
#   bash slurm/setup_env.sh
set -euo pipefail

# AJUSTE: rode `module avail python` e `module avail anaconda` primeiro
# pra achar o nome exato do módulo disponível no Santos Dumont -- os
# nomes variam por cluster/versão, não dá pra adivinhar aqui.
# module load anaconda3/<versao-que-existir>

ENV_DIR="$HOME/envs/pairs-rl"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Ambiente criado em $ENV_DIR"
echo "O .sbatch já ativa esse mesmo caminho -- se você mudar ENV_DIR aqui,  ajuste lá também."
