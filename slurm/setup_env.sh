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

# Venv em /scratch, NÃO em $HOME (/prj) -- descobrimos que o /prj tem
# lag de propagação entre nós (login/computo veem versões
# desatualizadas por um tempo depois de escrito), o que já quebrou um
# job tentando ativar um venv recém-criado ali. /scratch se mostrou
# consistente em todos os testes.
ENV_DIR="/scratch/ppg-lncc/$USER/envs/pairs-rl"
rm -rf "$ENV_DIR"  # remove venv antigo se existir
mkdir -p "/scratch/ppg-lncc/$USER/pairs-rl-logs"  # saida dos jobs tambem vai pro scratch, mesmo motivo

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip

# torch é dependência do stable-baselines3, mas o pip install normal
# baixa a build com CUDA completa (torch + nvidia-cudnn/nccl/cusparselt/
# etc, ~2GB) mesmo não tendo GPU nesses nós -- instala a build CPU-only
# primeiro, bem mais leve, pra "requirements.txt" abaixo reaproveitar
# em vez de baixar a versão GPU.
pip install "torch>=2.8,<3.0" --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

echo ""
echo "Ambiente criado em $ENV_DIR"
echo "O .sbatch já ativa esse mesmo caminho -- se você mudar ENV_DIR aqui,  ajuste lá também."
