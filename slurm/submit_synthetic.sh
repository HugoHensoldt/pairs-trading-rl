#!/bin/bash
# Experimento SINTÉTICO com estado "só preços" (STATE_KIND=raw), autônomo (roda no
# login node, sobrevive a logout com setsid+nohup):
#   1) teste rápido do estado raw e do gerador (segundos);
#   2) para KIND em coint, null: treina (submit_all_folds.sh, chunks de 20 min), depois
#      roda o diagnóstico (submit_diag.sbatch) e, no coint, a referência da regra ótima.
# coint = existe arbitragem (o agente deveria aprendê-la); null = não existe (controle
# negativo: o agente deve ficar quase flat e o alinhamento deve ser ~0).
# Não repete execuções cujo state.json já está "done": true. Uma execução que falha
# NÃO é repetida e o driver segue para a próxima.
#
# Uso (na raiz do repo):
#   DRY_RUN=1 bash slurm/submit_synthetic.sh      # só imprime o que faria
#   setsid nohup bash slurm/submit_synthetic.sh > /scratch/ppg-lncc/$USER/sintetico.log 2>&1 < /dev/null &
# Progresso: /scratch/ppg-lncc/$USER/sintetico_progresso.md
set -uo pipefail

KINDS="${KINDS:-coint null}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
PROGRESS="${PROGRESS:-/scratch/ppg-lncc/$USER/sintetico_progresso.md}"
cd "$(dirname "$0")/.."

# configuração comum (o que já estiver exportado tem precedência)
export STATE_KIND=raw ENV_KIND=hedged
export TRAIN_SIZES="${TRAIN_SIZES:-300}" N_VAL="${N_VAL:-30}" N_TEST="${N_TEST:-30}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000000}" ENT_COEF="${ENT_COEF:-0.01}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-5000000}" CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-5000000}"
export VEC_ENV="${VEC_ENV:-dummy}" TORCH_THREADS="${TORCH_THREADS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4096}" N_STEPS="${N_STEPS:-2048}"
export THROUGHPUT_STEPS_PER_SEC="${THROUGHPUT_STEPS_PER_SEC:-27000}" AVG_TICKS_PER_DAY=27000
export SYNTH_DAYS="${SYNTH_DAYS:-450}"

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"

mkdir -p "$(dirname "$PROGRESS")"
log() { echo "$@" | tee -a "$PROGRESS"; }
wait_queue() { while squeue --me --noheader 2>/dev/null | grep -q .; do sleep 10; done; }
run() { if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN: $*"; else "$@"; fi; }

log "##### SINTÉTICO: kinds=[$KINDS] seed=$SEED total=$TOTAL_TIMESTEPS treino=$TRAIN_SIZES val/teste=$N_VAL/$N_TEST — $(date) #####"

log "== teste rápido (raw + sintético)"
if [ "$DRY_RUN" != "1" ]; then
    if ! (cd src && python smoke_test_raw_synthetic.py) 2>&1 | tee -a "$PROGRESS" | tail -3 | grep -q "TESTES OK"; then
        log "ERRO: smoke_test_raw_synthetic.py falhou; abortando (veja o log)"; exit 1
    fi
fi

for kind in $KINDS; do
    tag="syn_${kind}_s${SEED}"
    state="src/runs/${tag}/fold1/state.json"
    if [ -f "$state" ] && grep -q '"done": true' "$state"; then
        log "[$(date)] $tag já concluída; pulando o treino"
    else
        log "##### [$(date)] INÍCIO treino $tag #####"
        if ! run env SYNTHETIC="$kind" SYNTH_KIND="$kind" RUN_TAG="$tag" SEED="$SEED" bash slurm/submit_all_folds.sh; then
            log "##### [$(date)] FALHOU $tag (não será repetida; segue para a próxima) #####"
            continue
        fi
        log "##### [$(date)] FIM treino $tag #####"
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: sbatch --wait diag de $tag"; continue
    fi
    wait_queue
    log "##### [$(date)] INÍCIO diagnóstico $tag #####"
    sbatch --wait --export=ALL,DIAG_RUN_DIR="runs/${tag}/fold1",DIAG_KIND="$kind" slurm/submit_diag.sbatch \
        || log "AVISO: o job de diagnóstico de $tag terminou com erro/timeout"
    log "##### [$(date)] FIM diagnóstico $tag #####"
    {
        echo; echo "### Resumo de $tag ($(date))"
        for f in src/runs/$tag/fold1/diag/summary.md src/runs/$tag/fold1/diag/benchmark_stdout.txt; do
            [ -f "$f" ] && { echo "--- $f"; cat "$f"; }
        done
    } >> "$PROGRESS"
done
log "##### SINTÉTICO concluído — $(date) #####"
