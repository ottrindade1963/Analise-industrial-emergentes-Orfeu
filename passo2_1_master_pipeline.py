"""Pipeline Mestre do Passo 2.1: Limpeza, Agregação e EDA dos Agregados."""
import os
import sys

def executar_passo2_1_completo():
    print("=" * 70)
    print("INICIANDO PASSO 2.1 COMPLETO: LIMPEZA, AGREGAÇÃO E EDA")
    print("=" * 70)
    
    # 1. Limpeza
    print("\n[1/3] EXECUTANDO LIMPEZA DE DADOS QUANTITATIVOS...")
    from passo2_1_limpeza_pipeline import executar_limpeza
    executar_limpeza()
    
    # 2. Agregação
    print("\n[2/3] EXECUTANDO AGREGAÇÃO (3 MÉTODOS)...")
    from passo2_1_agregacao_pipeline import executar_agregacao
    executar_agregacao()
    
    # 3. EDA Agregados
    print("\n[3/3] EXECUTANDO ANÁLISE EXPLORATÓRIA DOS AGREGADOS...")
    from passo2_1_eda_agreg_pipeline import executar_eda_agregados
    executar_eda_agregados()
    
    print("\n" + "=" * 70)
    print("PASSO 2.1 CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    executar_passo2_1_completo()
