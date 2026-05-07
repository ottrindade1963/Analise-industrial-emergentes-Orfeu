"""Pipeline de limpeza e tratamento de dados."""

from passo2_1_limpeza_config import DATA_PATH, OUTPUT_DIR
from passo2_1_limpeza_processor import (
    carregar_dados, remover_paises_incompletos, remover_linhas_incompletas,
    imputar_valores, validar_ranges, gerar_relatorio, salvar_dados
)


def executar_limpeza():
    """Executa o pipeline completo de limpeza."""
    
    print("=" * 60)
    print("  LIMPEZA E TRATAMENTO DE DADOS")
    print("  Países Emergentes — World Bank Indicators")
    print("=" * 60)
    
    # Carregar dados originais
    df_original = carregar_dados(DATA_PATH)
    df = df_original.copy()
    
    # Fase 1: Remover países muito incompletos
    df = remover_paises_incompletos(df)
    
    # Fase 2: Remover linhas muito incompletas
    df = remover_linhas_incompletas(df)
    
    # Fase 3: Imputar valores ausentes
    df = imputar_valores(df)
    
    # Fase 4: Validar ranges
    df = validar_ranges(df)
    
    # Gerar relatório
    stats = gerar_relatorio(df_original, df)
    
    # Salvar dados
    csv_path, xlsx_path = salvar_dados(df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("  ✅ LIMPEZA CONCLUÍDA COM SUCESSO!")
    print(f"  📁 Ficheiros gerados em: {OUTPUT_DIR}/")
    print("=" * 60)
    
    return df, stats

if __name__ == "__main__":
    executar_limpeza()
