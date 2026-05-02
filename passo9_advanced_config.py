import os

# Detectar ambiente (Colab vs Local)
if os.path.exists('/content'):
    # Estamos no Colab
    BASE_DIR = '/content/Analise-industrial-emergentes-Orfeu'
else:
    # Estamos localmente
    BASE_DIR = os.getcwd()

# Configurações de Caminhos
DATA_DIR = os.path.join(BASE_DIR, 'dadosgerados')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analises_avancadas')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Focar apenas nos melhores modelos para análises avançadas
MODELS = ['RandomForest', 'XGBoost']
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']

# Configurações de Sensibilidade
PERTURBATION_LEVELS = [-0.2, -0.1, 0.1, 0.2] # -20%, -10%, +10%, +20%
