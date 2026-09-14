#!/bin/bash
# Treina 1 fold do walk-forward. A conta ppg-lncc/QOS normal tem
# MaxSubmit=1 (só 1 job na fila por vez) -- por isso NÃO usamos
# --array aqui (um array de N tasks conta como N jobs simultâneos pro
# Slurm e é rejeitado com AssocMaxSubmitJobLimit). Os folds treinam
# sequencialmente, um job por vez, cada um ainda usando 1 nó/48 núcleos.
#
# Uso direto (roda o fold 1 por padrão):
#   sbatch slurm/submit_fold.sbatch
#
# Pra escolher o fold, ou pra rodar todos em sequência automaticamente,
# use slurm/submit_all_folds.sh em vez de chamar este arquivo direto.
#
# Acompanhar:
#   squeue --me
#   tail -f pairs-rl-fold<N>-<jobid>.out

#SBATCH --job-name=pairs-rl-fold1
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --account=ppg-lncc
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Se o cluster exigir QOS explícita (nem toda config de Slurm exige),
# descomente e ajuste:
# #SBATCH --qos=sequana_cpu_dev

set -euo pipefail

# --- ambiente Python ---------------------------------------------------
# mesmo módulo usado em slurm/setup_env.sh -- AJUSTE junto se mudar lá
module load python/3.10.16_sequana
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

# --- qual fold treinar --------------------------------------------------
# slurm/submit_all_folds.sh exporta SLURM_ARRAY_TASK_ID antes do sbatch;
# chamando este arquivo direto (sem exportar nada), cai no fold 1.
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# SLURM_SUBMIT_DIR = diretório de onde `sbatch` foi chamado -- rode
# `sbatch slurm/submit_fold.sbatch` a partir da raiz do repo
cd "$SLURM_SUBMIT_DIR/src"
srun python rl_trading_pipeline.py
