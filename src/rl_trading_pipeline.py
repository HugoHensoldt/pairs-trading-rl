"""
Reinforcement Learning para negociação intradiária BOVA11 x WINM21 --
VERSÃO MINIMALISTA: uma única feature (spread de mispricing).

Diferente da abordagem supervisionada (LSTM prevendo Lucro/Prejuízo por
negociação com stops fixos Re/Ri), aqui o agente controla a posição TICK A
TICK: decide entrar, segurar ou sair a cada instante, aprendendo sua
própria política de entrada/saída.

Nesta versão, o estado do agente é composto por UMA janela de N_TICKS
valores do spread de mispricing (preço atual do WIN menos preço justo
implícito pelo BOVA11) -- sem bandas de Bollinger, sem volume, sem
retornos separados. A ideia é verificar se essa única informação, com
histórico suficiente, já basta para o agente aprender uma política
lucrativa, antes de adicionar mais features.

Requer:
    pip install gymnasium stable-baselines3

Estrutura:
  1) process_day()          -> gera o preço-justo (Wajusto/Wbjusto) de um
                                pregão, sem bandas de Bollinger
  2) build_feature_matrix() -> calcula o spread de mispricing por tick
                                (única feature)
  3) fit_feature_scaler()   -> normaliza usando estatística SOMENTE do
                                treino
  4) ArbitrageTradingEnv    -> ambiente Gymnasium: 1 episódio = 1 pregão
  5) MultiDayEnv            -> alterna entre vários pregões pré-carregados
                                a cada reset, para treinar em múltiplos dias
  6) train_ppo()            -> treina um agente PPO (stable-baselines3)
  7) evaluate_policy()      -> roda a política treinada nos dias de
                                teste/validação e reporta lucro total,
                                número de negociações e taxa de acerto
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from config import data_path


# --------------------------------------------------------------------------
# 1) PREÇO-JUSTO (sem bandas, sem máquina de estados de posição)
# --------------------------------------------------------------------------

def process_day(order, Periodo=3000):
    """Gera preço-justo (Wajusto/Wbjusto) para um pregão. Não calcula mais
    bandas de Bollinger (miWa/miWb/StDa/StDb/BOLUC/BOLDV) -- a única
    feature usada agora é o spread de mispricing, derivado direto de
    Wajusto/Wbjusto.
    """
    with open(data_path(str(order) + 'BOVA11.json'), 'r', encoding="UTF-16 LE") as myfile:
        dfb = pd.read_json(myfile)
    with open(data_path(str(order) + 'WINM21.json'), 'r', encoding="UTF-16 LE") as WI:
        dfi = pd.read_json(WI)

    dfb.index = dfb["miliseconds"]
    dfb = dfb.rename(columns={'miliseconds': 'ms', 'ask': 'bask', 'bid': 'bbid',
                               'last': 'blast', 'volume': 'bvolume', 'flags': 'bflags'})
    dfi.index = dfi["miliseconds"]
    dfi = dfi.rename(columns={'miliseconds': 'ms'})

    ds = pd.merge_ordered(dfb, dfi, fill_method="ffill")
    ds['Cask_3k'] = (ds['ask'] / ds['bask']).rolling(window=Periodo).mean()
    ds['Cbid_3k'] = (ds['bid'] / ds['bbid']).rolling(window=Periodo).mean()
    ds['Wbjusto'] = ds['bbid'] * ds['Cbid_3k']
    ds['Wajusto'] = ds['bask'] * ds['Cask_3k']

    dj = ds.copy()
    dj.index = pd.to_datetime(dj["datahora"])
    dj = dj.between_time('10:20', '16:30')
    dj.index = dj["ms"]
    dj = dj[Periodo:len(dj)]

    df = dj.dropna().reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 2) FEATURE ÚNICA: SPREAD DE MISPRICING
# --------------------------------------------------------------------------

RL_FEATURE_NAMES = ['spread_mispricing']


def build_feature_matrix(df):
    """Constrói a matriz de features (n_ticks, 1) usada como observação.

    Feature única: spread de mispricing = ponto médio do preço atual do
    WIN menos o ponto médio do preço justo implícito pelo BOVA11.
        spread > 0  -> WIN "caro" em relação ao justo (viés de venda)
        spread < 0  -> WIN "barato" em relação ao justo (viés de compra)

    O agente recebe uma JANELA de N_TICKS valores desse spread (não só o
    valor instantâneo) -- é essa janela que permite à rede aprender algo
    equivalente a uma média/desvio móvel, mesmo sem bandas de Bollinger
    calculadas explicitamente.
    """
    price_mid = (df['ask'] + df['bid']) / 2
    fair_mid = (df['Wajusto'] + df['Wbjusto']) / 2
    spread = (price_mid - fair_mid).to_numpy(dtype=float)
    return spread.reshape(-1, 1)


def fit_feature_scaler(feature_matrices):
    """Ajusta média/desvio-padrão usando SOMENTE as matrizes de treino
    (lista de arrays, uma por dia). Retorna (mean, std) para normalizar
    qualquer outra matriz depois.
    """
    stacked = np.concatenate(feature_matrices, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0  # evita divisão por zero em colunas constantes
    return mean, std


def apply_scaler(feature_matrix, mean, std):
    return (feature_matrix - mean) / std


# --------------------------------------------------------------------------
# 3) AMBIENTE GYMNASIUM -- 1 EPISÓDIO = 1 PREGÃO
# --------------------------------------------------------------------------

class ArbitrageTradingEnv(gym.Env):
    """Ambiente de RL para um único pregão.

    Ação (Discrete(3)): posição-ALVO a assumir neste tick.
        0 = flat
        1 = comprado
        2 = vendido  (internamente convertido para -1)

    Observação (Box): janela de N_TICKS ticks de features relativas,
    achatada em um vetor, concatenada com [posição atual (one-hot, 3),
    PnL não-realizado normalizado (1)].

    Recompensa: variação de marcação a mercado (mid-price) da posição já
    aberta desde o tick anterior, MENOS custo de execução (metade do
    spread) sempre que uma perna abre/fecha, MENOS taxa fixa por
    negociação fechada (replicando o custo de R$3 do backtest original).
    """

    metadata = {"render_modes": []}

    def __init__(self, df, feature_matrix, n_ticks=120, transaction_fee=3.0,
                 reward_scale=1.0):
        super().__init__()
        assert len(df) == len(feature_matrix)
        self.df = df
        self.feature_matrix = feature_matrix
        self.n_ticks = n_ticks
        self.transaction_fee = transaction_fee
        self.reward_scale = reward_scale
        self.n_features = feature_matrix.shape[1]

        obs_dim = self.n_ticks * self.n_features + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.t = self.n_ticks - 1          # primeiro tick com histórico completo
        self.position = 0                  # -1, 0, 1
        self.entry_price = 0.0             # preço de execução da posição aberta
        self.n_trades_closed = 0
        self.n_trades_won = 0
        self.trade_pnls = []               # P&L realizado de CADA negócio fechado

    def _get_obs(self):
        window = self.feature_matrix[self.t - self.n_ticks + 1: self.t + 1]
        window_flat = window.flatten()

        pos_onehot = np.zeros(3, dtype=np.float32)
        pos_onehot[self.position + 1] = 1.0  # posição -1,0,1 -> índice 0,1,2

        bid_t = self.df['bid'].iloc[self.t]
        ask_t = self.df['ask'].iloc[self.t]
        if self.position == 1:
            unrealized = bid_t - self.entry_price
        elif self.position == -1:
            unrealized = self.entry_price - ask_t
        else:
            unrealized = 0.0
        # normalização grosseira pelo spread médio para manter escala razoável
        unrealized_norm = np.array([unrealized / max(ask_t - bid_t, 1e-6)],
                                    dtype=np.float32)

        return np.concatenate([window_flat, pos_onehot, unrealized_norm]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _close_leg(self, bid_t, ask_t):
        """Fecha a posição aberta no tick atual, retornando (reward_da_saida,
        lucro_realizado). Aplica meio-spread de execução + taxa fixa, e
        registra o P&L líquido (já descontado desse custo) em
        self.trade_pnls para diagnóstico posterior.
        """
        if self.position == 1:
            exec_price = bid_t             # vende no bid para fechar compra
            realized = exec_price - self.entry_price
            spread_cost = (ask_t - bid_t) / 2
        elif self.position == -1:
            exec_price = ask_t             # compra no ask para fechar venda
            realized = self.entry_price - exec_price
            spread_cost = (ask_t - bid_t) / 2
        else:
            return 0.0, 0.0

        reward = -spread_cost - self.transaction_fee
        net_realized = realized - spread_cost - self.transaction_fee
        self.n_trades_closed += 1
        self.trade_pnls.append(net_realized)
        if realized > 0:
            self.n_trades_won += 1
        return reward, realized

    def _open_leg(self, target_position, bid_t, ask_t):
        """Abre uma nova posição no tick atual. Retorna a recompensa
        (custo de execução de meio-spread)."""
        if target_position == 1:
            self.entry_price = ask_t       # compra no ask
            spread_cost = (ask_t - bid_t) / 2
        elif target_position == -1:
            self.entry_price = bid_t       # vende no bid
            spread_cost = (ask_t - bid_t) / 2
        else:
            self.entry_price = 0.0
            spread_cost = 0.0
        self.position = target_position
        return -spread_cost

    def step(self, action):
        target_position = {0: 0, 1: 1, 2: -1}[int(action)]

        bid_t = self.df['bid'].iloc[self.t]
        ask_t = self.df['ask'].iloc[self.t]
        mid_t = (bid_t + ask_t) / 2

        reward = 0.0

        # 1) marcação a mercado da posição já aberta ANTES desta ação,
        #    desde o mid-price do tick anterior até agora
        if self.t > self.n_ticks - 1:
            bid_prev = self.df['bid'].iloc[self.t - 1]
            ask_prev = self.df['ask'].iloc[self.t - 1]
            mid_prev = (bid_prev + ask_prev) / 2
            reward += self.position * (mid_t - mid_prev)

        # 2) troca de posição: fecha a perna atual (se houver) e abre a nova
        #    (se houver) -- cobre flat->long, flat->short, long->short, etc.
        if target_position != self.position:
            close_reward, _ = self._close_leg(bid_t, ask_t)
            reward += close_reward
            reward += self._open_leg(target_position, bid_t, ask_t)

        self.t += 1
        terminated = self.t >= len(self.df) - 1
        truncated = False

        # 3) fim do pregão: força o fechamento de qualquer posição aberta
        if terminated and self.position != 0:
            bid_last = self.df['bid'].iloc[self.t]
            ask_last = self.df['ask'].iloc[self.t]
            close_reward, _ = self._close_leg(bid_last, ask_last)
            reward += close_reward
            self.position = 0

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32)

        info = {
            "n_trades_closed": self.n_trades_closed,
            "n_trades_won": self.n_trades_won,
        }
        return obs, reward * self.reward_scale, terminated, truncated, info


# --------------------------------------------------------------------------
# 4) AMBIENTE MULTI-DIA -- alterna entre pregões pré-carregados a cada reset
# --------------------------------------------------------------------------

class MultiDayEnv(gym.Env):
    """Envolve vários ArbitrageTradingEnv (um por pregão) e sorteia um
    pregão diferente a cada reset(), para o agente treinar em vários dias
    sem recarregar/reprocessar os JSONs a cada episódio.
    """

    metadata = {"render_modes": []}

    def __init__(self, day_envs, seed=None):
        super().__init__()
        assert len(day_envs) > 0
        self.day_envs = day_envs
        self.observation_space = day_envs[0].observation_space
        self.action_space = day_envs[0].action_space
        self.rng = np.random.default_rng(seed)
        self.current_env = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_env = self.day_envs[self.rng.integers(len(self.day_envs))]
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)


def build_day_envs(orders, mean, std, n_ticks=120, transaction_fee=3.0):
    """Carrega e processa cada pregão em `orders` UMA VEZ, retornando uma
    lista de ArbitrageTradingEnv prontos para uso -- evita reler os JSONs
    a cada episódio de treino.
    """
    envs = []
    for order in orders:
        df = process_day(order)
        feat = build_feature_matrix(df)
        feat = apply_scaler(feat, mean, std)
        envs.append(ArbitrageTradingEnv(df, feat, n_ticks=n_ticks,
                                         transaction_fee=transaction_fee))
    return envs


# --------------------------------------------------------------------------
# 5) TREINO (PPO)
# --------------------------------------------------------------------------

def train_ppo(train_orders, val_orders, n_ticks=120, total_timesteps=5000, #300_000,
              model_path="ppo_arbitrage.zip"):
    # a normalização é ajustada SOMENTE com os dias de treino
    train_dfs_feats = []
    for order in train_orders:
        df = process_day(order)
        train_dfs_feats.append(build_feature_matrix(df))
    mean, std = fit_feature_scaler(train_dfs_feats)

    train_envs = build_day_envs(train_orders, mean, std, n_ticks=n_ticks)
    val_envs = build_day_envs(val_orders, mean, std, n_ticks=n_ticks)

    def make_train_env():
        return Monitor(MultiDayEnv(train_envs))

    def make_val_env():
        return Monitor(MultiDayEnv(val_envs))

    vec_train_env = DummyVecEnv([make_train_env])
    vec_val_env = DummyVecEnv([make_val_env])

    model = PPO(
        "MlpPolicy",
        vec_train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,        # horizonte longo: episódio = um pregão inteiro
        verbose=1,
    )

    eval_callback = EvalCallback(
        vec_val_env, best_model_save_path="./best_model",
        eval_freq=10_000, n_eval_episodes=len(val_orders), deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(model_path)

    return model, (mean, std)


# --------------------------------------------------------------------------
# 6) AVALIAÇÃO -- roda a política determinística nos dias de teste
# --------------------------------------------------------------------------

def evaluate_policy(model, orders, mean, std, n_ticks=120):
    results = []
    all_trade_pnls = []  # P&L de TODOS os negócios, de todos os dias, para o diagnóstico

    for order in orders:
        df = process_day(order)
        feat = apply_scaler(build_feature_matrix(df), mean, std)
        env = ArbitrageTradingEnv(df, feat, n_ticks=n_ticks)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results.append({
            "order": order,
            "lucro_total": total_reward,
            "negocios_fechados": info["n_trades_closed"],
            "negocios_vencedores": info["n_trades_won"],
        })
        all_trade_pnls.extend(env.trade_pnls)

    results_df = pd.DataFrame(results)
    print(results_df)
    print(f"\nLucro total agregado: {results_df['lucro_total'].sum():.2f}")
    print(f"Negócios fechados: {results_df['negocios_fechados'].sum()}")
    print(f"Taxa de acerto: "
          f"{results_df['negocios_vencedores'].sum() / max(results_df['negocios_fechados'].sum(), 1):.2%}")

    # --- Diagnóstico de distribuição de P&L por negócio (já líquido de custos) ---
    pnls = np.array(all_trade_pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    print("\n=== Distribuição de P&L por negócio (líquida de custos) ===")
    print(f"Total de negócios: {len(pnls)}  |  vencedores: {len(wins)}  |  perdedores: {len(losses)}")
    if len(wins) > 0:
        print(f"Ganho médio       : {wins.mean():.2f}   |  mediana: {np.median(wins):.2f}   |  máximo: {wins.max():.2f}")
    if len(losses) > 0:
        print(f"Perda média       : {losses.mean():.2f}   |  mediana: {np.median(losses):.2f}   |  mínimo: {losses.min():.2f}")
    if len(wins) > 0 and len(losses) > 0:
        razao = abs(losses.mean()) / wins.mean()
        print(f"Perda média / Ganho médio: {razao:.2f}x")
    # percentis extremos ajudam a enxergar a "cauda" de perdas grandes
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  percentil {p:>2}: {np.percentile(pnls, p):.2f}")

    return results_df, pnls


# --------------------------------------------------------------------------
# EXECUÇÃO
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Dynamic split for train, validation, and test sets
    all_orders = list(range(1, 339)) # Example: adjust this range as needed
    total_days = len(all_orders)

    train_split = int(total_days * 0.70)
    val_split = int(total_days * 0.15)

    train_orders = all_orders[:train_split]
    val_orders = all_orders[train_split:train_split + val_split]
    test_orders = all_orders[train_split + val_split:]

    print("Treino:", train_orders)
    print("Validação:", val_orders)
    print("Teste:", test_orders)

    model, (mean, std) = train_ppo(train_orders, val_orders, total_timesteps=300_000)

    print("\n=== Avaliação em VALIDAÇÃO ===")
    evaluate_policy(model, val_orders, mean, std)

    print("\n=== Avaliação em TESTE ===")
    evaluate_policy(model, test_orders, mean, std)
