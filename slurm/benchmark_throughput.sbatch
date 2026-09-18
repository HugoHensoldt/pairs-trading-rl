#!/bin/bash
# Mede o throughput REAL de treino (steps/s) da v4 no nó do Santos Dumont.
# Rode ANTES da rodada completa e ajuste THROUGHPUT_STEPS_PER_SEC /
# TOTAL_TIMESTEPS / N_STEPS com o resultado.
#
# Uso (a partir da raiz do repo em /scratch):
#   sbatch slurm/benchmark_throughput.sbatch
#   N_STEPS=4096 sbatch slurm/benchmark_throughput.sbatch   # compara n_steps
# Saída: /scratch/ppg-lncc/$USER/pairs-rl-logs/pairs-rl-bench-<jobid>.out

#SBATCH --job-name=pairs-rl-bench
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:15:00
#SBATCH --account=ppg-lncc
#SBATCH --output=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.out
#SBATCH --error=/scratch/ppg-lncc/%u/pairs-rl-logs/%x-%j.err

set -euo pipefail
module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"
export MODE="${MODE:-both}"

cd "$SLURM_SUBMIT_DIR/src"
srun python benchmark_env_throughput.py
