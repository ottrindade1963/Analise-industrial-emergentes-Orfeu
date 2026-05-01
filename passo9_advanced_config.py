import os

# Configurações de Caminhos
BASE_DIR = '/content/Analise-industrial-emergentes-Orfeu'
DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analises_avancadas')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Análise de Sensibilidade
SENSITIVITY_VARS = ['wgi_control_corruption', 'wgi_rule_law', 'fbc_percent_pib']
SENSITIVITY_STEPS = [-0.2, -0.1, 0.0, 0.1, 0.2] # Variações percentuais (ex: -20% a +20%)

# Datasets e Modelos a analisar (foco nos melhores)
BEST_DATASET = 'inner'
BEST_STRATEGY = 'A3_Interacao'
BEST_MODELS = ['RandomForest', 'XGBoost']
