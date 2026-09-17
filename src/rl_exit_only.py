"""
Terceira abordagem de RL para BOVA11 x WINM21: um agente cujo ÚNICO papel
é decidir QUANDO SAIR de uma arbitragem já aberta.

Ao contrário de rl_trading_pipeline.py (agente controla entrada, permanência
e saída tick a tick) e de lstm_pipeline.py (rede supervisionada decide ANTES
de entrar se aceita ou recusa a oportunidade), aqui:

  - a ENTRADA é decidida pela MESMA lógica de sinal do backtest original
    (bandas de Bollinger sobre o preço-justo, ver backtest_verify.process_day
    -- reaproveitada por IMPORT, não reimplementada);
  - o agente não escolhe direção nem se entra -- só tem uma decisão BINÁRIA
    a cada tick depois da entrada: continuar segurando (0) ou sair agora (1);
  - a saída fixa por Re/Ri do backtest original é IGNORADA (é exatamente o
    que o agente aprende a substituir); a única saída não-aprendida é o
    fechamento forçado no fim do pregão.

Por que isso é comparável às outras duas abordagens: as três partem do MESMO
conjunto de negociações sinalizadas pelo backtest (mesma detecção de
transição flat->posição usada em lstm_pipeline.extract_trade_windows). A
LSTM decide ANTES de entrar (aceita/recusa); este agente decide DEPOIS de
já ter entrado (quando sair). Comparar o lucro agregado das três dá uma
resposta direta a "aprender o momento certo de sair recupera mais lucro do
que aprender quais entradas aceitar?".

Estrutura:
  1) extract_trade_entries()  -> localiza, no df do backtest original, cada
                                  transição flat->posição (mesmo método do
                                  lstm_pipeline) e devolve entrada/direção
  2) build_trade_dataset()    -> processa vários pregões, monta a matriz de
                                  features (reaproveitada de
                                  rl_trading_pipeline.build_feature_matrix)
                                  e a lista de negociações candidatas
  3) ExitOnlyEnv              -> ambiente Gymnasium: 1 episódio = 1 negociação
                                  já aberta, ação Discrete(2) (segurar/sair)
  4) MultiTradeEnv            -> alterna entre negociações pré-carregadas a
                                  cada reset, para treinar em várias de uma vez
  5) train_ppo()               -> treina um agente PPO (stable-baselines3)
  6) evaluate_policy()         -> roda a política nas negociações de
                                  teste/validação e compara o lucro obtido
                                  contra o lucro que a regra fixa Re/Ri do
                                  backtest original teria dado NAS MESMAS
                                  negociações

Requer:
    pip install gymnasium stable-baselines3
"""

import os

# Mesmo motivo de rl_trading_pipeline.py: em cluster (Santos Dumont),
# n_train_envs sobe para ~48 processos paralelos (SubprocVecEnv, um por
# núcleo físico do nó) -- sem isso, cada processo abriria suas próprias
# threads BLAS e o total estouraria muito além dos núcleos disponíveis.
# Precisa ser setado ANTES de importar numpy/torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from functools import partial

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from config import POINT_VALUE_BRL
from backtest_verify import process_day as backtest_process_day
from rl_trading_pipeline import (
    build_feature_matrix,
    RL_FEATURE_NAMES,
    fit_feature_scaler,
    apply_scaler,
    _detect_n_train_envs,
)

# mesmo motivo de torch.set_num_threads(1) em rl_trading_pipeline.py: o
# processo PRINCIPAL também roda forward/backward via torch, concorrendo
# por CPU com os processos do SubprocVecEnv.
torch.set_num_threads(1)

# --------------------------------------------------------------------------
# 1) LOCALIZA AS NEGOCIAÇÕES SINALIZADAS PELO BACKTEST ORIGINAL
# --------------------------------------------------------------------------


def extract_trade_entries(df):
    """Localiza cada transição flat->posição (0->1 ou 0->-1) no df já
    processado por backtest_process_day(). Mesmo critério usado em
    lstm_pipeline.extract_trade_windows (posicao, não 'p' -- 'p' pode
    continuar != 0 em vários ticks com a posição já aberta e geraria
    entradas duplicadas para a mesma negociação).

    Devolve uma lista de dicts (um por negociação candidata):
        entry_pos    -> posição (iloc) do tick de entrada
        direction    -> +1 (comprado) ou -1 (vendido)
        entry_price  -> preço de execução na entrada (ask p/ compra, bid p/
                         venda) -- igual ao que o backtest original usou
        baseline_lucro -> 'lucro' realizado pela regra fixa Re/Ri do
                         backtest original NESSA MESMA negociação (só para
                         comparação/relatório -- o agente NUNCA vê isso)

    `df` deve ter índice posicional (reset_index(drop=True)) -- ver
    build_trade_dataset().
    """
    buy_entry = (df['posicao'].shift(1) == 0) & (df['posicao'] == 1)
    sell_entry = (df['posicao'].shift(1) == 0) & (df['posicao'] == -1)
    entry_mask = buy_entry | sell_entry

    trades = []
    for idx in df.index[entry_mask]:
        pos = df.index.get_loc(idx)
        direction = 1 if df.loc[idx, 'posicao'] == 1 else -1
        entry_price = float(df.loc[idx, 'entrada'])

        # lucro que a regra fixa Re/Ri teria realizado nesta negociação --
        # só para comparação depois de avaliar a política, nunca entra na
        # observação nem na recompensa do agente.
        close_idx = df.index[(df.index > idx) & (df['posicao'] == 0)]
        baseline_lucro = float(df.loc[close_idx[0], 'lucro']) if len(close_idx) else None

        trades.append({
            "entry_pos": pos,
            "direction": direction,
            "entry_price": entry_price,
            "baseline_lucro": baseline_lucro,
        })
    return trades


# --------------------------------------------------------------------------
# 2) MONTA O DATASET (features por pregão + negociações candidatas)
# --------------------------------------------------------------------------

def build_trade_dataset(orders):
    """Processa cada pregão em `orders` UMA VEZ: roda o backtest original
    (backtest_process_day, mesma lógica de sinal de config.py/
    backtest_verify.py) e monta a matriz de features tick a tick
    (reaproveitada de rl_trading_pipeline.build_feature_matrix -- as
    colunas 'ask'/'bid'/'Wajusto'/'Wbjusto' de que ela precisa já existem
    no df do backtest original, então não há reimplementação).

    Retorna (days, trades):
      days   -> lista de dicts {"order", "df", "feature_matrix"} (índice
                posicional, um por pregão)
      trades -> lista de dicts (ver extract_trade_entries) + "day_idx"
                (índice em `days` a que a negociação pertence)
    """
    days = []
    trades = []
    for order in orders:
        df = backtest_process_day(order).reset_index(drop=True)
        feat = build_feature_matrix(df)

        day_idx = len(days)
        days.append({"order": order, "df": df, "feature_matrix": feat})

        for trade in extract_trade_entries(df):
            trade["day_idx"] = day_idx
            trade["order"] = order
            trades.append(trade)

    return days, trades


# --------------------------------------------------------------------------
# 3) AMBIENTE GYMNASIUM -- 1 EPISÓDIO = 1 NEGOCIAÇÃO JÁ ABERTA
# --------------------------------------------------------------------------

N_TICKS = 60  # janela de observação (features pré-normalizadas, ver escala.py)
TRANSACTION_FEE = 2.5  # mesma convenção de rl_trading_pipeline.py (2.5 pontos == R$0,50)


class ExitOnlyEnv(gym.Env):
    """Um episódio = uma negociação já sinalizada e ABERTA pelo backtest
    original. O agente só decide, a cada tick, entre:
        0 = continuar segurando
        1 = sair agora

    Não há ação de entrar nem de escolher direção -- `direction` e
    `entry_price` vêm fixos da detecção de sinal (extract_trade_entries).

    Observação: janela de N_TICKS ticks das 5 features de
    rl_trading_pipeline.build_feature_matrix (mesmas usadas na abordagem de
    RL completo, já normalizadas -- ver apply_scaler), achatada, concatenada
    com [direção (+1/-1), ticks em posição (normalizado), PnL não-realizado
    (normalizado pelo spread)]. Ticks antes do início do pregão (janela
    incompleta perto da entrada) são preenchidos com zero.

    Recompensa: mesma convenção de custos de rl_trading_pipeline.py --
    marcação a mercado tick a tick da posição, menos meio-spread de
    execução na entrada E na saída, menos taxa fixa cobrada inteira na
    entrada. A soma da recompensa ao longo do episódio é exatamente o P&L
    líquido realizado dessa negociação -- comparável direto ao 'lucro' do
    backtest original e ao lucro por negócio da abordagem de RL completo.
    """

    metadata = {"render_modes": []}

    def __init__(self, df, feature_matrix, entry_pos, direction, entry_price,
                 n_ticks=N_TICKS, transaction_fee=TRANSACTION_FEE,
                 reward_scale=1.0, max_hold_ticks=None):
        super().__init__()
        self.bid = df['bid'].to_numpy(dtype=float)
        self.ask = df['ask'].to_numpy(dtype=float)
        self.feature_matrix = feature_matrix
        self.n_ticks = n_ticks
        self.transaction_fee = transaction_fee
        self.reward_scale = reward_scale
        # limite de segurança opcional (regra FIXA, não aprendida) -- desligado
        # por padrão, igual ao max_loss_per_position de rl_trading_pipeline.
        self.max_hold_ticks = max_hold_ticks
        self.n_features = feature_matrix.shape[1]

        self.entry_pos = entry_pos
        self.direction = direction
        self.entry_price = entry_price
        self.last_tick = len(self.bid) - 1

        obs_dim = self.n_ticks * self.n_features + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._reset_state()

    def _reset_state(self):
        self.t = self.entry_pos
        self.ticks_held = 0
        self.done = False
        self._entry_cost_pending = True
        self.net_pnl = None
        self.exited_voluntarily = None

    def _get_obs(self):
        start = self.t - self.n_ticks + 1
        if start < 0:
            window = self.feature_matrix[0:self.t + 1]
            pad = np.zeros((-start, self.n_features), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        else:
            window = self.feature_matrix[start:self.t + 1]
        window_flat = window.flatten()

        bid_t, ask_t = self.bid[self.t], self.ask[self.t]
        if self.direction == 1:
            unrealized = bid_t - self.entry_price
        else:
            unrealized = self.entry_price - ask_t
        unrealized_norm = unrealized / max(ask_t - bid_t, 1e-6)

        ticks_norm = self.ticks_held / 1000.0

        extra = np.array([self.direction, ticks_norm, unrealized_norm], dtype=np.float32)
        return np.concatenate([window_flat, extra]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _close(self, bid_t, ask_t):
        """Fecha a posição no tick atual. `realized` (exec_price -
        entry_price) já embute os dois meios-spreads (entrada e saída)
        implicitamente, porque entry_price/exec_price usam ask/bid em vez
        do mid-price -- por isso `net_realized` não subtrai spread_cost de
        novo, e a taxa fixa só é descontada aqui (a cobrança de
        transaction_fee em step() afeta só `reward`, não este diagnóstico;
        ver mesma lógica/comentário em rl_trading_pipeline._close_leg)."""
        if self.direction == 1:
            exec_price = bid_t
            realized = exec_price - self.entry_price
        else:
            exec_price = ask_t
            realized = self.entry_price - exec_price
        spread_cost = (ask_t - bid_t) / 2
        net_realized = realized - self.transaction_fee
        return -spread_cost, net_realized

    def step(self, action):
        assert not self.done, "step() chamado depois do episódio terminar -- chame reset()"

        bid_t, ask_t = self.bid[self.t], self.ask[self.t]
        mid_t = (bid_t + ask_t) / 2
        reward = 0.0

        if self._entry_cost_pending:
            # taxa fixa + meio-spread de execução na entrada, cobrados uma
            # única vez -- mesma convenção de _open_leg em rl_trading_pipeline.
            entry_spread_cost = (self.ask[self.entry_pos] - self.bid[self.entry_pos]) / 2
            reward += -entry_spread_cost - self.transaction_fee
            self._entry_cost_pending = False
        elif self.t > self.entry_pos:
            mid_prev = (self.bid[self.t - 1] + self.ask[self.t - 1]) / 2
            reward += self.direction * (mid_t - mid_prev)

        force_exit = (self.t >= self.last_tick) or \
            (self.max_hold_ticks is not None and self.ticks_held >= self.max_hold_ticks)
        exit_now = bool(action) or force_exit

        terminated = False
        net_pnl_total = None
        if exit_now:
            close_reward, net_realized = self._close(bid_t, ask_t)
            reward += close_reward
            terminated = True
            self.exited_voluntarily = bool(action) and not force_exit
            net_pnl_total = net_realized
        else:
            self.t += 1
            self.ticks_held += 1

        self.done = terminated
        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32)

        info = {}
        if terminated:
            info = {
                "net_pnl": net_pnl_total,
                "ticks_held": self.ticks_held,
                "exited_voluntarily": self.exited_voluntarily,
            }

        return obs, reward * self.reward_scale, terminated, False, info


# --------------------------------------------------------------------------
# 4) AMBIENTE MULTI-NEGOCIAÇÃO -- sorteia uma negociação pré-carregada por reset
# --------------------------------------------------------------------------

class MultiTradeEnv(gym.Env):
    """Envolve várias negociações candidatas (uma ExitOnlyEnv por
    negociação) e sorteia uma diferente a cada reset(), análogo a
    MultiDayEnv em rl_trading_pipeline.py mas em granularidade de
    NEGOCIAÇÃO em vez de PREGÃO inteiro.
    """

    metadata = {"render_modes": []}

    def __init__(self, days, trades, n_ticks=N_TICKS,
                 transaction_fee=TRANSACTION_FEE, max_hold_ticks=None, seed=None):
        super().__init__()
        assert len(trades) > 0, "nenhuma negociação candidata -- verifique os pregões usados"
        self.trade_envs = [
            ExitOnlyEnv(
                days[tr["day_idx"]]["df"], days[tr["day_idx"]]["feature_matrix"],
                tr["entry_pos"], tr["direction"], tr["entry_price"],
                n_ticks=n_ticks, transaction_fee=transaction_fee,
                max_hold_ticks=max_hold_ticks,
            )
            for tr in trades
        ]
        self.observation_space = self.trade_envs[0].observation_space
        self.action_space = self.trade_envs[0].action_space
        self.rng = np.random.default_rng(seed)
        self.current_env = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_env = self.trade_envs[self.rng.integers(len(self.trade_envs))]
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)


def _make_multitrade_env(orders, mean, std, n_ticks=N_TICKS, max_hold_ticks=None):
    """Função construtora usada como env_fn do SubprocVecEnv -- mesmo
    padrão de _make_multiday_env em rl_trading_pipeline.py: recebe só
    identificadores leves (orders + mean/std da normalização) e faz o
    trabalho pesado (reprocessar os pregões, montar features) DENTRO do
    processo em que é chamada, para não serializar DataFrames/arrays já
    montados pelo pipe do processo pai para cada processo-filho.
    """
    days, trades = build_trade_dataset(orders)
    for d in days:
        d["feature_matrix"] = apply_scaler(d["feature_matrix"], mean, std)
    return Monitor(MultiTradeEnv(days, trades, n_ticks=n_ticks, max_hold_ticks=max_hold_ticks))


# --------------------------------------------------------------------------
# 5) TREINO (PPO)
# --------------------------------------------------------------------------

def train_ppo(train_orders, val_orders, n_ticks=N_TICKS, total_timesteps=200_000,
              model_path="ppo_exit_only.zip", best_model_dir="./best_model_exit_only",
              max_hold_ticks=None):
    # a normalização é ajustada SOMENTE com os pregões de treino -- processa
    # uma vez aqui no processo principal (também serve pra contar quantas
    # negociações candidatas existem, log abaixo); os processos-filho do
    # SubprocVecEnv reprocessam os MESMOS train_orders de forma independente
    # (ver _make_multitrade_env), recebendo só mean/std pelo pickle.
    train_days, train_trades = build_trade_dataset(train_orders)
    mean, std = fit_feature_scaler([d["feature_matrix"] for d in train_days])

    val_days, val_trades = build_trade_dataset(val_orders)
    for d in val_days:
        d["feature_matrix"] = apply_scaler(d["feature_matrix"], mean, std)

    n_train_envs = _detect_n_train_envs()
    print(f"Negociações candidatas -- treino: {len(train_trades)}  validação: {len(val_trades)}  "
          f"(n_train_envs={n_train_envs})")

    train_env_fns = [
        partial(_make_multitrade_env, train_orders, mean, std, n_ticks, max_hold_ticks)
        for _ in range(n_train_envs)
    ]
    vec_train_env = SubprocVecEnv(train_env_fns)

    def make_val_env():
        return Monitor(MultiTradeEnv(val_days, val_trades, n_ticks=n_ticks,
                                      max_hold_ticks=max_hold_ticks))

    vec_val_env = DummyVecEnv([make_val_env])

    model = PPO(
        "MlpPolicy",
        vec_train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,
        verbose=1,
    )

    # eval_freq é contado em chamadas de callback (uma por env.step() do
    # VecEnv), não em timesteps totais -- cada chamada avança n_train_envs
    # timesteps de uma vez (todos os processos em paralelo dão 1 passo por
    # chamada). Mesmo raciocínio de rl_trading_pipeline.train_ppo: queremos
    # ~18 avaliações ao longo do treino inteiro.
    n_avaliacoes_alvo = 18
    eval_freq = max(1, int(total_timesteps / (n_avaliacoes_alvo * n_train_envs)))

    eval_callback = EvalCallback(
        vec_val_env, best_model_save_path=best_model_dir,
        eval_freq=eval_freq,
        n_eval_episodes=min(200, len(val_trades)),
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(model_path)

    return model, (mean, std)


# --------------------------------------------------------------------------
# 6) AVALIAÇÃO -- roda a política determinística em cada negociação de teste
#    e compara contra o que a regra fixa Re/Ri do backtest original faria
# --------------------------------------------------------------------------

def evaluate_policy(model, orders, mean, std, n_ticks=N_TICKS, max_hold_ticks=None):
    days, trades = build_trade_dataset(orders)
    for d in days:
        d["feature_matrix"] = apply_scaler(d["feature_matrix"], mean, std)

    results = []
    for tr in trades:
        d = days[tr["day_idx"]]
        env = ExitOnlyEnv(d["df"], d["feature_matrix"], tr["entry_pos"],
                           tr["direction"], tr["entry_price"],
                           n_ticks=n_ticks, max_hold_ticks=max_hold_ticks)
        obs, _ = env.reset()
        terminated = False
        info = {}
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        results.append({
            "order": tr["order"],
            "direction": tr["direction"],
            "agent_pnl": info["net_pnl"],
            "ticks_held": info["ticks_held"],
            "exited_voluntarily": info["exited_voluntarily"],
            "baseline_pnl": tr["baseline_lucro"],
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    agent_total = results_df['agent_pnl'].sum()
    baseline_total = results_df['baseline_pnl'].sum()
    n_trades = len(results_df)
    n_won = (results_df['agent_pnl'] > 0).sum()

    print(f"\n=== Comparação: agente de saída vs. regra fixa Re/Ri (mesmas "
          f"{n_trades} negociações) ===")
    print(f"Lucro agente   : {agent_total:.2f} pontos (R$ {agent_total * POINT_VALUE_BRL:.2f})")
    print(f"Lucro regra fixa: {baseline_total:.2f} pontos "
          f"(R$ {baseline_total * POINT_VALUE_BRL:.2f})")
    print(f"Diferença      : {(agent_total - baseline_total):.2f} pontos")
    print(f"Taxa de acerto do agente: {n_won / max(n_trades, 1):.2%}")
    print(f"Ticks médios em posição : {results_df['ticks_held'].mean():.1f}")

    return results_df


# --------------------------------------------------------------------------
# EXECUÇÃO
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    all_orders = list(range(1, 339))
    max_pregoes = os.environ.get("MAX_PREGOES")
    if max_pregoes:
        all_orders = all_orders[:int(max_pregoes)]

    total_days = len(all_orders)
    train_split = int(total_days * 0.70)
    val_split = int(total_days * 0.15)

    train_orders = all_orders[:train_split]
    val_orders = all_orders[train_split:train_split + val_split]
    test_orders = all_orders[train_split + val_split:]
    print("Treino:", train_orders)
    print("Validação:", val_orders)
    print("Teste:", test_orders)

    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", "200000"))
    model, (mean, std) = train_ppo(train_orders, val_orders, total_timesteps=total_timesteps)

    print("\n=== Avaliação em VALIDAÇÃO ===")
    evaluate_policy(model, val_orders, mean, std)

    print("\n=== Avaliação em TESTE ===")
    evaluate_policy(model, test_orders, mean, std)
