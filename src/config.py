"""
Configuração compartilhada pelos scripts do projeto.

Se você mover a pasta de dados no futuro, só precisa mudar o caminho
AQUI -- os três scripts (backtest_verify, lstm_pipeline,
rl_trading_pipeline) importam DATA_DIR deste arquivo.
"""

import os
from pathlib import Path

# Pasta onde ficam os arquivos {numero}BOVA11.json e {numero}WINM21.json.
# Lida da env var TICK_DATA_DIR quando setada (ex.: no .sbatch do
# Santos Dumont, apontando pro scratch) -- senão cai no caminho local do
# WSL usado em desenvolvimento. Use "r" antes da string (raw string) no
# Windows para não precisar escapar as barras invertidas.
DATA_DIR = Path(os.environ.get("TICK_DATA_DIR", r"/mnt/c/Users/HugoV/tick_data"))


def data_path(filename: str) -> Path:
    """Monta o caminho completo de um arquivo dentro de DATA_DIR."""
    return DATA_DIR / filename
