#!/bin/bash
# Submete os folds da v4 em sequência, um job por vez -- necessário porque
# a conta ppg-lncc/QOS normal tem MaxSubmit=1 (só 1 job na fila por vez; um
# --array de N tasks conta como N jobs e é rejeitado).
#
# Para CADA fold:
#   1) encadeia jobs de TREINO (chunks de ~20 min, checkpoint + resume) até
#      o state.json do fold marcar "done": true (total_timesteps atingido);
#   2) submete 2 jobs só de AVALIAÇÃO (RUN_EVAL=1 e 2): melhor-de-validação e
#      último modelo em val/teste, baselines, sensibilidade a custo, teste
#      de latência, diagnóstico de arbitragem e curva de checkpoints -- tudo
#      em CSV em src/runs/<RUN_TAG>/fold<N>/eval/.
#
# O número de chunks NÃO é fixo: o pipeline treina até total_timesteps (ver
# TOTAL_TIMESTEPS) e o fim é decidido pelo state.json. O plano estimado
# (PRINT_FOLD_PLAN) só serve de referência e de teto (2x + 2) contra loop
# infinito.
#
# Uso:
#   bash slurm/submit_all_folds.sh
#   FOLDS="1" bash slurm/submit_all_folds.sh              # só o fold 1
#   START_CHUNK=4 FOLDS="2" bash slurm/submit_all_folds.sh # retoma o fold 2 no chunk 4
#
# Env vars repassadas ao pipeline (todas opcionais): TOTAL_TIMESTEPS,
# TRAIN_SIZES, N_VAL, N_TEST, GAMMA, N_STEPS, BATCH_SIZE, SEED, RUN_TAG,
# THROUGHPUT_STEPS_PER_SEC, MAX_PREGOES -- ver __main__ em
# src/rl_trading_pipeline.py.
set -euo pipefail

N_FOLDS=3  # tamanhos de treino em TRAIN_SIZES (default 100,150,200) -- 1 fold cada

export RUN_TAG="${RUN_TAG:-v4}"
FOLDS="${FOLDS:-$(seq 1 "$N_FOLDS")}"
START_CHUNK="${START_CHUNK:-1}"

cd "$(dirname "$0")/.."

# O plano roda no login node (fora do sbatch): ativa o mesmo venv/módulo.
module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"

echo "=== Plano estimado de chunks por fold (RUN_TAG=$RUN_TAG, FOLDS=$FOLDS) ==="
declare -A CHUNKS_POR_FOLD
while read -r fold_num n_chunks; do
    CHUNKS_POR_FOLD["$fold_num"]="$n_chunks"
    echo "  Fold $fold_num: ~$n_chunks chunk(s) (estimativa)"
done < <(cd src && PRINT_FOLD_PLAN=1 python3 rl_trading_pipeline.py | grep -E '^[0-9]+ [0-9]+$')

wait_for_empty_queue() {
    while squeue --me --noheader --format="%j" 2>/dev/null | grep -q '^pairs-rl-fold'; do
        echo "Aguardando fila liberar (ainda há job pairs-rl-fold* ativo)..."
        sleep 5
    done
}

state_done() {   # $1 = state.json
    [ -f "$1" ] && grep -q '"done": true' "$1"
}

n_chunks_done() {   # $1 = state.json
    if [ -f "$1" ]; then grep -c '"chunk":' "$1" || true; else echo 0; fi
}

for i in $FOLDS; do
    estimado="${CHUNKS_POR_FOLD[$i]:?fold $i não apareceu no plano do PRINT_FOLD_PLAN}"
    max_chunks=$(( estimado * 2 + 2 ))
    state="src/runs/${RUN_TAG}/fold${i}/state.json"

    chunk="$START_CHUNK"
    while [ "$chunk" -le "$max_chunks" ]; do
        if [ "$chunk" -gt 1 ] && state_done "$state"; then
            break
        fi
        resume=1
        if [ "$chunk" -eq 1 ]; then resume=0; fi

        wait_for_empty_queue
        echo "=== Fold $i (RUN_TAG=$RUN_TAG), chunk $chunk (estimado ~$estimado) ($(date)) ==="
        sbatch --wait \
            --job-name="pairs-rl-fold${i}-${RUN_TAG}" \
            --export=ALL,SLURM_ARRAY_TASK_ID="$i",RESUME_TRAINING="$resume",CHUNK_INDEX="$chunk",RUN_EVAL=0 \
            slurm/submit_fold.sbatch || echo "AVISO: job do fold $i chunk $chunk terminou com erro/timeout"
        echo "=== Fold $i, chunk $chunk concluído ($(date)) ==="

        # se o job morreu sem gravar o chunk no state.json, pára em vez de
        # repetir o mesmo chunk indefinidamente
        if [ "$(n_chunks_done "$state")" -lt "$chunk" ]; then
            echo "ERRO: chunk $chunk do fold $i não foi registrado em $state -- abortando." >&2
            echo "Veja os logs em /scratch/ppg-lncc/$USER/pairs-rl-logs/ e retome com START_CHUNK=$chunk FOLDS=$i." >&2
            exit 1
        fi
        chunk=$(( chunk + 1 ))
    done

    if ! state_done "$state"; then
        echo "AVISO: fold $i não atingiu TOTAL_TIMESTEPS em $max_chunks chunks -- avaliando o que existe."
    fi

    # avaliação em 2 jobs (juntos passam de 20 min):
    #   parte 1 = modelos principais + baselines + sensibilidade a custo
    #   parte 2 = teste de latência + curva de checkpoints
    for part in 1 2; do
        wait_for_empty_queue
        echo "=== Fold $i: job de avaliação parte $part ($(date)) ==="
        sbatch --wait \
            --job-name="pairs-rl-fold${i}-${RUN_TAG}-eval${part}" \
            --export=ALL,SLURM_ARRAY_TASK_ID="$i",RESUME_TRAINING=1,CHUNK_INDEX=0,RUN_EVAL="$part" \
            slurm/submit_fold.sbatch || echo "AVISO: avaliação (parte $part) do fold $i terminou com erro/timeout"
    done
    echo "=== Fold $i concluído ($(date)) ==="
done

echo "=== Folds concluídos: $FOLDS ==="
