"""
Smoke test temporário: confirma que a refatoração .iloc -> arrays numpy
em ArbitrageTradingEnv não mudou o comportamento do ambiente.

Roda um episódio inteiro com uma sequência de ações determinística
(seed fixa) e imprime um resumo (soma de recompensas, hash da sequência
de recompensas, número de trades) que pode ser comparado entre a versão
antes e depois da refatoração.

Uso:
    python src/smoke_test_env_equivalence.py
"""

import hashlib

import numpy as np

from rl_trading_pipeline import process_day, build_feature_matrix, ArbitrageTradingEnv

ORDER = 1
N_TICKS = 120
N_STEPS = 5000


def main():
    df = process_day(ORDER)
    feat = build_feature_matrix(df)
    env = ArbitrageTradingEnv(df, feat, n_ticks=N_TICKS)

    obs, _ = env.reset()
    rng = np.random.default_rng(42)

    rewards = []
    for _ in range(N_STEPS):
        action = int(rng.integers(0, 3))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            obs, _ = env.reset()

    rewards = np.array(rewards, dtype=np.float64)
    reward_hash = hashlib.sha256(rewards.tobytes()).hexdigest()

    print(f"Steps rodados     : {len(rewards)}")
    print(f"Soma de recompensas: {rewards.sum():.10f}")
    print(f"Hash das recompensas: {reward_hash}")
    print(f"n_trades_closed final: {info['n_trades_closed']}")
    print(f"n_trades_won final   : {info['n_trades_won']}")


if __name__ == "__main__":
    main()
