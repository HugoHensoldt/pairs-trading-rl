"""
Benchmark de throughput do ArbitrageTradingEnv.

Script standalone (não altera o pipeline principal). Constrói o ambiente
para 1-2 pregões de exemplo, roda um número fixo de steps com ações
aleatórias medindo o tempo de parede, e imprime steps/segundo.

Com base no throughput medido, estima quantos total_timesteps cabem em
uma janela de treino de N horas -- considerando o paralelismo que o
ambiente JÁ suporta hoje (verificado no pipeline: train_ppo() usa
DummyVecEnv([make_train_env]) com uma única função de ambiente, ou seja,
1 ambiente único, sem paralelismo de processos -- ver
rl_trading_pipeline.py). O DummyVecEnv roda os ambientes sequencialmente
na mesma thread, então o throughput agregado hoje é simplesmente o
throughput de 1 ambiente.

Uso:
    python src/benchmark_env_throughput.py
"""

import time

import numpy as np

from rl_trading_pipeline import (
    process_day,
    build_feature_matrix,
    fit_feature_scaler,
    apply_scaler,
    ArbitrageTradingEnv,
)

N_STEPS = 20_000
BENCHMARK_ORDERS = [1, 2]          # pregões de exemplo
N_TICKS = 120                      # mesmo default usado em train_ppo()
TRAINING_HOURS = 6.0
N_PARALLEL_ENVS_TODAY = 1          # DummyVecEnv([make_train_env]) -> 1 env, sem SubprocVecEnv


def build_benchmark_env(orders, n_ticks=N_TICKS):
    """Carrega os pregões de exemplo e monta um único ArbitrageTradingEnv
    concatenando-os um após o outro (suficiente para medir custo por
    step; não precisamos do MultiDayEnv/sorteio para este benchmark)."""
    feats = []
    dfs = []
    for order in orders:
        df = process_day(order)
        feats.append(build_feature_matrix(df))
        dfs.append(df)

    mean, std = fit_feature_scaler(feats)

    # usa só o primeiro pregão carregado para o loop de steps -- se ele
    # não tiver ticks suficientes para N_STEPS, o benchmark reseta o
    # ambiente (novo episódio) e continua contando steps
    df0 = dfs[0]
    feat0 = apply_scaler(feats[0], mean, std)
    env = ArbitrageTradingEnv(df0, feat0, n_ticks=n_ticks)
    return env


def run_benchmark():
    env = build_benchmark_env(BENCHMARK_ORDERS)

    obs, _ = env.reset()
    rng = np.random.default_rng(0)

    start = time.perf_counter()
    for _ in range(N_STEPS):
        action = rng.integers(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - start

    steps_per_sec = N_STEPS / elapsed
    return steps_per_sec, elapsed


def main():
    print(f"Rodando benchmark: {N_STEPS} steps com ações aleatórias "
          f"(pregões de exemplo: {BENCHMARK_ORDERS})...")
    steps_per_sec, elapsed = run_benchmark()

    print(f"\nTempo total       : {elapsed:.2f} s")
    print(f"Steps/segundo     : {steps_per_sec:,.1f}")

    print(f"\nParalelismo hoje  : {N_PARALLEL_ENVS_TODAY} ambiente "
          f"(DummyVecEnv com uma única função de ambiente -- sem "
          f"SubprocVecEnv no pipeline atual)")

    total_seconds = TRAINING_HOURS * 3600
    estimated_timesteps = steps_per_sec * N_PARALLEL_ENVS_TODAY * total_seconds

    print(f"\nEstimativa para {TRAINING_HOURS:.0f}h de treino "
          f"(i5 11ª geração, paralelismo atual = {N_PARALLEL_ENVS_TODAY}):")
    print(f"  total_timesteps ~= {estimated_timesteps:,.0f}")
    print("\n(Nota: esta estimativa conta só o custo do ambiente/step. O "
          "tempo real de treino também inclui forward/backward do PPO, "
          "que não é medido aqui.)")


if __name__ == "__main__":
    main()
