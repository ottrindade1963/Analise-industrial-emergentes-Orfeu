import os

# Configurações de Caminhos
# Detecção automática de ambiente (Colab vs Local)
if os.path.exists('/content'):
    BASE_DIR = '/content/Analise-industrial-emergentes-Orfeu'
else:
    # Se não estiver no Colab, usa o diretório atual
    BASE_DIR = os.getcwd()

# Agora os dados vêm diretamente das pastas de saída do Passo 2.1
OUTPUT_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nomes dos Ficheiros de Entrada (Agora incluindo o Não Agregado)
DATASETS = {
    'nao_agregado': os.path.join(BASE_DIR, 'dados_limpos', 'wdi_emergentes_limpo.csv'),
    'inner': os.path.join(BASE_DIR, 'agregado_metodo1_inner', 'agregado_inner.csv'),
    'left': os.path.join(BASE_DIR, 'agregado_metodo2_left_imputado', 'agregado_left_imputado.csv'),
    'outer': os.path.join(BASE_DIR, 'agregado_metodo3_outer_completo', 'agregado_outer_completo.csv')
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
    'wgi_voice_accountability'
]

# Variáveis Quantitativas (para interações)
QUANTITATIVE_VARS_FOR_INTERACTION = [
    'formacao_bruta_capital_fixo_percent_pib', # Formação Bruta de Capital Fixo
    'investimento_estrangeiro_direto_percent_pib', # Investimento Estrangeiro Direto
    'comercio_percent_pib' # Abertura Comercial
]
