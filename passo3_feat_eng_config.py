import os

# Configurações de Caminhos
BASE_DIR = '/content/Analise-industrial-emergentes-Orfeu'
DATA_DIR = os.path.join(BASE_DIR, 'dadospreparados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'dados_engenharia')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nomes dos Ficheiros de Entrada (Agora incluindo o Não Agregado)
DATASETS = {
    'nao_agregado': 'wdi_emergentes_limpo.csv',
    'inner': 'agregado_inner.csv',
    'left': 'agregado_left_imputado.csv',
    'outer': 'agregado_outer_completo.csv'
}

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Variáveis Qualitativas (Institucionais e Política)
QUALITATIVE_VARS = [
    'wgi_control_corruption', 
    'wgi_gov_effectiveness', 
    'wgi_political_stability', 
    'wgi_regulatory_quality', 
    'wgi_rule_law', 
    'wgi_voice_accountability', 
    'icrg_qog'
]

# Variáveis Quantitativas (para interações)
# Assumindo nomes comuns do WDI, ajustar conforme necessário
QUANTITATIVE_VARS_FOR_INTERACTION = [
    'fbc_percent_pib', # Formação Bruta de Capital Fixo
    'ide_percent_pib', # Investimento Estrangeiro Direto
    'comercio_percent_pib' # Abertura Comercial
]
