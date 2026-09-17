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
#   tail -f /scratch/ppg-lncc/$USER/pairs-rl-logs/pairs-rl-fold<N>-<jobid>.out

#SBATCH --job-name=pairs-rl-fold1
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --account=ppg-lncc
# Logs em /scratch, nao no diretorio do repo (/prj) -- /prj tem lag de
# propagacao entre nos que ja causou falha (ver job 11594210: nao viu o
# venv recem-criado). %u/%x/%j sao placeholders do proprio Slurm.
#SBATCH --output=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.out
#SBATCH --error=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.err

# Se o cluster exigir QOS explícita (nem toda config de Slurm exige),
# descomente e ajuste:
# #SBATCH --qos=sequana_cpu_dev

set -euo pipefail

# --- ambiente Python ---------------------------------------------------
# mesmo módulo/caminho usados em slurm/setup_env.sh -- AJUSTE junto se
# mudar lá. Venv em /scratch (não /prj), ver comentário no --output acima.
module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"

# --- dados ---------------------------------------------------------------
# AJUSTE: caminho real depois de copiar os dados pro scratch (ver
# comando scp/rsync sugerido na conversa) -- confirme o nome do
# diretório de scratch do seu projeto com `ls /scratch/`
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"

# --- qual fold treinar / treino em chunks (resume) ------------------------
# slurm/submit_all_folds.sh exporta SLURM_ARRAY_TASK_ID (fold) e, quando o
# fold precisa de mais de 1 chunk pra chegar em TARGET_N_PASSADAS (ver
# generate_walk_forward_folds), também RESUME_TRAINING/FINAL_CHUNK.
# Chamando este arquivo direto (sem exportar nada), os defaults preservam
# o comportamento de sempre: fold 1, treina do zero, avalia ao final.
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
export RESUME_TRAINING="${RESUME_TRAINING:-0}"
export FINAL_CHUNK="${FINAL_CHUNK:-1}"
# RUN_TAG/CHUNK_INDEX: ver slurm/submit_all_folds.sh -- defaults preservam
# o comportamento de sempre (v1/v2) quando este arquivo é chamado direto.
export RUN_TAG="${RUN_TAG:-}"
export CHUNK_INDEX="${CHUNK_INDEX:-1}"

# Escolha de hoje (2026-09-16): usar bem menos que os 480 pregões
# disponíveis (ver MAX_PREGOES em rl_trading_pipeline.py) -- mesmo default
# usado em slurm/submit_all_folds.sh, repetido aqui pra quem chamar este
# arquivo direto (sem passar por submit_all_folds.sh) também ficar na
# mesma escala reduzida em vez de cair nos 480 pregões completos.
export MAX_PREGOES="${MAX_PREGOES:-80}"

# --- margem de segurança sob o MaxTime=00:20:00 (1200s) do job ------------
# Contados a partir do início do processo Python (SCRIPT_START), não do
# início do learn() -- ver TimeLimitCallback em rl_trading_pipeline.py.
#
# JOB_HARD_SECONDS=1170 (19min30, 30s de colchão antes do kill de
# verdade): se não sobrar pelo menos EVAL_RESERVE_SECONDS (150s, ver
# rl_trading_pipeline.py) até essa marca, a avaliação de
# validação/teste é PULADA em vez de arriscar ser cortada no meio (o
# checkpoint do modelo já está salvo nesse ponto, então nada se perde
# de fato -- só o relatório impresso, recuperável depois).
#
# TRAIN_MAX_SECONDS varia conforme FINAL_CHUNK (ver slurm/submit_all_folds.sh
# pra como isso é orquestrado num treino em vários chunks encadeados):
# - chunk final (FINAL_CHUNK=1, ou não setado -- job único como antes):
#   780s (13min), reservando tempo pro evaluate_policy() de validação+
#   teste que roda depois do learn().
# - chunk intermediário (FINAL_CHUNK=0): não faz avaliação, então pode
#   treinar bem mais perto do limite de 20min -- 1140s (19min), só
#   reservando tempo pro model.save() final e encerramento do job.
#
# IMPORTANTE: os 1140s do chunk intermediário são o mesmo valor do
# default `chunk_seconds` em generate_walk_forward_folds()
# (rl_trading_pipeline.py) -- é isso que converte total_timesteps em
# n_chunks_needed por fold. Mudar um sem o outro faz o número de chunks
# planejado (PRINT_FOLD_PLAN) não bater com o tempo real disponível por
# job.
#
# Um smoke test real (fold 1, 7 dias de treino) mostrou ~180s de
# overhead antes do learn() comecar e a avaliação sozinha não terminou
# em ~90s -- estes valores dão bastante margem sobre isso. Folds com
# mais dias de treino (ex.: fold 3, 45 dias) tendem a ter overhead de
# carregamento de dados MAIOR, não menor -- se a folga ainda não for
# suficiente, aumente esses números.
export JOB_HARD_SECONDS=1170
if [ "$FINAL_CHUNK" = "1" ]; then
    export TRAIN_MAX_SECONDS=780
else
    export TRAIN_MAX_SECONDS=1140
fi

# SLURM_SUBMIT_DIR = diretório de onde `sbatch` foi chamado -- rode
# `sbatch slurm/submit_fold.sbatch` a partir da raiz do repo clonado em
# /scratch (não em /prj -- ver comentário no --output acima).
cd "$SLURM_SUBMIT_DIR/src"
srun python rl_trading_pipeline.py
