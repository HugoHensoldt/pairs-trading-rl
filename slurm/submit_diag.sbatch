#!/bin/bash
# Diagnóstico de arbitragem de UMA execução (ver src/diagnose_agent.py) e, se for o
# sintético cointegrado, a referência da regra ótima (src/synthetic_benchmark.py).
#
# Variáveis (exporte com --export=ALL,...):
#   DIAG_RUN_DIR   pasta da execução relativa a src/ (ex.: runs/syn_coint_s0/fold1)
#   DIAG_KIND      coint | null | real   (define SYNTH_KIND e se roda a referência)
#   DIAG_ARGS      argumentos extras do diagnose_agent.py (default: val e teste, 30 pregões)
# Saída: <DIAG_RUN_DIR>/diag/*.csv, summary.md e diag_stdout.txt

#SBATCH --job-name=pairs-rl-diag
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --account=ppg-lncc
#SBATCH --output=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.out
#SBATCH --error=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.err

set -uo pipefail
module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"
export EVAL_WORKERS=48
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

: "${DIAG_RUN_DIR:?defina DIAG_RUN_DIR}"
DIAG_KIND="${DIAG_KIND:-real}"
DIAG_ARGS="${DIAG_ARGS:---sets val,test --max-days 30 --tests A,B,C,D}"
if [ "$DIAG_KIND" != "real" ]; then export SYNTH_KIND="$DIAG_KIND"; fi

cd "$SLURM_SUBMIT_DIR/src"
mkdir -p "$DIAG_RUN_DIR/diag"
echo "== diagnose_agent.py $DIAG_ARGS  ($(date))"
# shellcheck disable=SC2086
python diagnose_agent.py --run-dir "$DIAG_RUN_DIR" $DIAG_ARGS 2>&1 | tee "$DIAG_RUN_DIR/diag/diag_stdout.txt"

if [ "$DIAG_KIND" = "coint" ]; then
    echo "== synthetic_benchmark.py  ($(date))"
    python synthetic_benchmark.py --run-dir "$DIAG_RUN_DIR" 2>&1 | tee "$DIAG_RUN_DIR/diag/benchmark_stdout.txt"
fi
echo "== fim ($(date))"
