import os

# Configurações de Caminhos
BASE_DIR = '/content/Analise-industrial-emergentes-Orfeu'
DATA_DIR = os.path.join(BASE_DIR, 'dadosgerados') # Lê os dados gerados no Passo 3
OUTPUT_DIR = os.path.join(BASE_DIR, 'modelos_treinados')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Validação Cruzada (Walk-Forward Temporal por Ano)
# Considerando dados de 1995 a 2023 (29 anos)
TRAIN_END_YEAR = 2014 # 1995-2014 (20 anos = ~69%)
VAL_END_YEAR = 2017   # 2015-2017 (3 anos = ~10%)
# TEST: 2018-2023 (6 anos = ~21%)

# Modelos a serem treinados (Agora os 5 modelos do plano)
MODELS_TO_TRAIN = ['RandomForest', 'XGBoost', 'SARIMAX', 'LSTM', 'TFT']

# Datasets e Estratégias a processar (Incluindo Não Agregado)
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']

# Hiperparâmetros Básicos (para demonstração)
RF_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
XGB_PARAMS = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
LSTM_PARAMS = {'epochs': 50, 'batch_size': 32, 'units': 64}
TFT_PARAMS = {'epochs': 50, 'batch_size': 32, 'hidden_size': 32}
