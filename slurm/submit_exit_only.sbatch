#!/bin/bash
# Treina a abordagem de "agente de saída" (src/rl_exit_only.py) no Santos
# Dumont -- primeira rodada de teste: mede throughput real antes de
# decidir se vale a pena portar a infra de chunking/resume/walk-forward
# CV de submit_fold.sbatch pra cá (ver docs/relatorio_resultados*.md pro
# histórico dessa infra na abordagem de RL completo).
#
# Diferente de submit_fold.sbatch: episódios aqui são por NEGOCIAÇÃO
# (algumas centenas de ticks), não por PREGÃO inteiro (~30k ticks) -- a
# expectativa é que treine bem mais rápido, cabendo inteiro num job da
# partição de desenvolvimento sem precisar de checkpoint/resume. Se
# TOTAL_TIMESTEPS/MAX_PREGOES forem aumentados e o job passar a estourar
# --time, val a pena portar TimeLimitCallback + chunking (ver
# TimeLimitCallback em rl_trading_pipeline.py) antes de tentar rodadas
# maiores -- SEM isso, um job morto pelo Slurm no meio do treino perde
# tudo (nenhum checkpoint intermediário é salvo).
#
# Uso:
#   sbatch slurm/submit_exit_only.sbatch
#
# Acompanhar:
#   squeue --me
#   tail -f /scratch/ppg-lncc/$USER/pairs-rl-logs/pairs-rl-exit-only-<jobid>.out

#SBATCH --job-name=pairs-rl-exit-only
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --account=ppg-lncc
#SBATCH --output=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.out
#SBATCH --error=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.err

# Se o cluster exigir QOS explícita (nem toda config de Slurm exige),
# descomente e ajuste:
# #SBATCH --qos=sequana_cpu_dev

set -euo pipefail

# --- ambiente Python -------------------------------------------------------
# mesmo módulo/venv de slurm/setup_env.sh/submit_fold.sbatch -- reaproveita
# o MESMO ambiente já criado pra abordagem de RL completo (rl_exit_only.py
# usa só dependências já em requirements.txt, nenhuma instalação extra).
module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"

# --- dados -------------------------------------------------------------
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"

# --- escala da rodada de teste ---------------------------------------------
# Conservador de propósito (rodada de medição, não a rodada final) -- ver
# comentário no topo sobre TimeLimitCallback. Ajuste depois de ver o
# tempo real de parede desta primeira rodada.
export MAX_PREGOES="${MAX_PREGOES:-80}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"

cd "$SLURM_SUBMIT_DIR/src"
srun python rl_exit_only.py
