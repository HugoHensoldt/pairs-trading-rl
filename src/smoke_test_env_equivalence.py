"""
Smoke test do ArbitrageTradingEnv (v4: estado de 4 dimensões, só o tick atual).

Roda ações aleatórias determinísticas (seed fixa) e confere:
  - observation_space.shape == (4,) e observações finitas;
  - posição na observação em {-1, 0, +1};
  - soma das recompensas de um episódio == env.total_reward e
    custos (gross_mtm - total_reward) >= 0.
Imprime também o hash das recompensas. ATENÇÃO: como o episódio agora
começa no tick 0 (antes era no 119, por causa da janela), o hash NÃO é
comparável com o das versões v1-v3.

Uso:
    python src/smoke_test_env_equivalence.py
"""

import hashlib

import numpy as np

from rl_trading_pipeline import (
    process_day, build_state_features, fit_feature_scaler, apply_scaler, ArbitrageTradingEnv,
)

ORDER = 1
N_STEPS = 60_000  # > 2 pregões, para exercitar terminação e reset


def main():
    df = process_day(ORDER)
    feat_raw = build_state_features(df)
    mean, std = fit_feature_scaler([feat_raw])
    feat = apply_scaler(feat_raw, mean, std)
    env = ArbitrageTradingEnv(df, feat)
    assert env.observation_space.shape == (4,), env.observation_space.shape

    obs, _ = env.reset()
    assert obs.shape == (4,) and np.all(np.isfinite(obs)), obs
    rng = np.random.default_rng(42)

    rewards = []
    ep_reward = 0.0
    for _ in range(N_STEPS):
        action = int(rng.integers(0, 3))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        ep_reward += reward
        if not (terminated or truncated):
            assert np.all(np.isfinite(obs)) and obs[2] in (-1.0, 0.0, 1.0), obs
        if terminated or truncated:
            assert abs(ep_reward - env.total_reward) < 1e-6, (ep_reward, env.total_reward)
            assert env.gross_mtm - env.total_reward >= -1e-6, "custos negativos?"
            obs, _ = env.reset()
            ep_reward = 0.0

    rewards = np.array(rewards, dtype=np.float64)
    reward_hash = hashlib.sha256(rewards.tobytes()).hexdigest()

    print(f"Steps rodados     : {len(rewards)}")
    print(f"Soma de recompensas: {rewards.sum():.10f}")
    print(f"Hash das recompensas: {reward_hash}")
    print(f"n_trades_closed final: {info['n_trades_closed']}")
    print(f"n_trades_won final   : {info['n_trades_won']}")
    print(f"Shape da observação : {env.observation_space.shape}")
    print("OK: observações finitas, posição em {-1,0,1}, contabilidade consistente.")


if __name__ == "__main__":
    main()
