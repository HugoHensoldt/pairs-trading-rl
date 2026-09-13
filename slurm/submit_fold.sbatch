#!/bin/bash
# Treina 1 fold do walk-forward por task do array -- os N_FOLDS folds
# (hoje 3, ver N_FOLDS em rl_trading_pipeline.py) treinam em paralelo,
# cada um em 1 nó de 48 núcleos, em vez de em sequência num processo só.
#
# Uso:
#   sbatch slurm/submit_fold.sbatch
#
# Acompanhar:
#   squeue --me
#   tail -f slurm-<jobid>_<taskid>.out

#SBATCH --job-name=pairs-rl
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --account=ppg-lncc
#SBATCH --array=1-3
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# Se o cluster exigir QOS explícita (nem toda config de Slurm exige),
# descomente e ajuste:
# #SBATCH --qos=sequana_cpu_dev

set -euo pipefail

# --- ambiente Python ---------------------------------------------------
# mesmo módulo usado em slurm/setup_env.sh -- AJUSTE junto se mudar lá
# module load anaconda3/<versao-que-existir>
source "$HOME/envs/pairs-rl/bin/activate"

# --- dados ---------------------------------------------------------------
# AJUSTE: caminho real depois de copiar os dados pro scratch (ver
# comando scp/rsync sugerido na conversa) -- confirme o nome do
# diretório de scratch do seu projeto com `ls /scratch/`
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"

# --- margem de segurança sob o MaxTime=00:20:00 do job --------------------
# 1020s = 17min: para o model.learn() 3min antes do kill do Slurm, dando
# tempo de sobra pro model.save() terminar antes do hard limit.
export TRAIN_MAX_SECONDS=1020

# SLURM_SUBMIT_DIR = diretório de onde `sbatch` foi chamado -- rode
# `sbatch slurm/submit_fold.sbatch` a partir da raiz do repo
cd "$SLURM_SUBMIT_DIR/src"
srun python rl_trading_pipeline.py
