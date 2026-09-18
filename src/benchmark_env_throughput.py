"""
Benchmark de throughput da v4 (estado de 4 dimensões).

Dois números:
  1) `env`: steps/s do ArbitrageTradingEnv sozinho (1 processo, ações
     aleatórias) -- só o custo do ambiente.
  2) `ppo`: steps/s REAIS de treino (SubprocVecEnv com N processos + rede +
     atualizações do PPO) -- é o que dimensiona TOTAL_TIMESTEPS e o número de
     chunks. Esse número precisa ser medido no NÓ do Santos Dumont (ver
     slurm/benchmark_throughput.sbatch); o valor do PC doméstico não vale
     (o default antigo, 4209, foi medido no PC e o cluster deu ~5,5k na v3).

Uso:
    python src/benchmark_env_throughput.py            # roda os dois
    MODE=ppo N_STEPS=2048 python src/benchmark_env_throughput.py
    MODE=env python src/benchmark_env_throughput.py

Env vars: MODE (env|ppo|both), VEC_ENV (dummy|subproc, default dummy), TORCH_THREADS, N_TRAIN_ENVS, N_STEPS/BATCH_SIZE/... (os
mesmos hiperparâmetros PPO do pipeline), BENCH_DAYS (dias carregados por
processo, default 5), BENCH_ROLLOUTS (rollouts medidos, default 4).
"""

import os
import time
from functools import partial

import numpy as np

from rl_trading_pipeline import (
    process_day_cached,
    build_state_features,
    fit_feature_scaler,
    apply_scaler,
    ArbitrageTradingEnv,
    _make_multiday_env,
    _detect_n_train_envs,
    ppo_hparams_from_env,
)

N_STEPS_ENV = 60_000
BENCH_DAYS = int(os.environ.get("BENCH_DAYS", "5"))
BENCH_ROLLOUTS = int(os.environ.get("BENCH_ROLLOUTS", "4"))


def run_env_benchmark():
    df = process_day_cached(1)
    raw = build_state_features(df)
    mean, std = fit_feature_scaler([raw])
    env = ArbitrageTradingEnv(df, apply_scaler(raw, mean, std))
    env.reset()
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(N_STEPS_ENV):
        _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 3)))
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    return N_STEPS_ENV / elapsed


def run_ppo_benchmark():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    n_envs = int(os.environ.get("N_TRAIN_ENVS") or _detect_n_train_envs())
    orders = list(range(1, 1 + BENCH_DAYS))
    mean, std = fit_feature_scaler([build_state_features(process_day_cached(o)) for o in orders])
    hp = ppo_hparams_from_env()

    t0 = time.perf_counter()
    vec_cls = DummyVecEnv if os.environ.get("VEC_ENV", "dummy") == "dummy" else SubprocVecEnv
    vec = vec_cls([partial(_make_multiday_env, orders, mean, std, 2.5, None, i)
                   for i in range(n_envs)])
    model = PPO("MlpPolicy", vec, seed=0, verbose=0, **hp)
    startup = time.perf_counter() - t0

    per_rollout = hp["n_steps"] * n_envs
    model.learn(total_timesteps=per_rollout)  # aquecimento (não medido)
    t1 = time.perf_counter()
    model.learn(total_timesteps=per_rollout * BENCH_ROLLOUTS, reset_num_timesteps=False)
    elapsed = time.perf_counter() - t1
    vec.close()
    return n_envs, per_rollout, startup, per_rollout * BENCH_ROLLOUTS / elapsed, hp


def main():
    mode = os.environ.get("MODE", "both")
    if mode in ("env", "both"):
        print(f"[env]  {run_env_benchmark():,.0f} steps/s (1 processo, sem rede)")
    if mode in ("ppo", "both"):
        n_envs, per_rollout, startup, sps, hp = run_ppo_benchmark()
        print(f"[ppo]  {sps:,.0f} steps/s reais de treino com {n_envs} processos "
              f"(rollout de {per_rollout:,} steps, startup {startup:.0f}s)")
        print(f"       hiperparâmetros: {hp}")
        for total in (20e6, 30e6, 40e6):
            secs = total / sps
            print(f"       {total/1e6:.0f}M timesteps ≈ {secs/60:.0f} min de treino "
                  f"≈ {secs/1050:.1f} chunks de ~1050s efetivos")


if __name__ == "__main__":
    main()
