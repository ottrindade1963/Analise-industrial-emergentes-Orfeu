import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import passo3_feat_eng_config as config

class FeatureEngineer:
    def __init__(self, df, dataset_name):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.qual_vars = [col for col in config.QUALITATIVE_VARS if col in self.df.columns]
        
        # Identificar variáveis quantitativas disponíveis para interação
        self.quant_vars = [col for col in config.QUANTITATIVE_VARS_FOR_INTERACTION if col in self.df.columns]
        
        # Se não encontrar as variáveis exatas, tenta encontrar algumas numéricas para demonstração
        if not self.quant_vars:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in self.qual_vars and c != config.TARGET_VAR and c not in ['year', 'ano', 'pais', 'country']]
            self.quant_vars = numeric_cols[:3] if numeric_cols else []

    def apply_strategy_a1(self):
        """
        Estratégia A1: Inclusão Direta
        As variáveis qualitativas já estão no dataset, apenas garantimos que estão padronizadas.
        Para o dataset não agregado, apenas retorna o dataset original.
        """
        df_a1 = self.df.copy()
        
        # Padronizar variáveis qualitativas se existirem (datasets agregados)
        if self.qual_vars:
            scaler = StandardScaler()
            # Preencher NaNs com a média antes de padronizar
            df_a1[self.qual_vars] = df_a1[self.qual_vars].fillna(df_a1[self.qual_vars].mean())
            df_a1[self.qual_vars] = scaler.fit_transform(df_a1[self.qual_vars])
            
        return df_a1

    def apply_strategy_a2(self):
        """
        Estratégia A2: Fator Latente (PCA)
        Extrai o primeiro componente principal das variáveis institucionais.
        Para o dataset não agregado, retorna o dataset original.
        """
        df_a2 = self.df.copy()
        
        if len(self.qual_vars) > 1:
            # Preencher possíveis NaNs (caso existam) com a média para o PCA
            pca_data = df_a2[self.qual_vars].fillna(df_a2[self.qual_vars].mean())
            
            # Padronizar antes do PCA
            scaler = StandardScaler()
            pca_data_scaled = scaler.fit_transform(pca_data)
            
            # Aplicar PCA (1 componente)
            pca = PCA(n_components=1)
            fator_institucional = pca.fit_transform(pca_data_scaled)
            
            # Adicionar o fator latente
            df_a2['fator_institucional_pca'] = fator_institucional
            
            # Remover as variáveis originais para reduzir dimensionalidade
            df_a2 = df_a2.drop(columns=self.qual_vars)
            
            print(f"[{self.dataset_name}] PCA Variância Explicada: {pca.explained_variance_ratio_[0]:.2%}")
            
        return df_a2

    def apply_strategy_a3(self):
        """
        Estratégia A3: Termos de Interação
        Cria produtos entre variáveis qualitativas e quantitativas selecionadas.
        Para o dataset não agregado, retorna o dataset original.
        """
        df_a3 = self.apply_strategy_a1() # Começa com A1 (padronizado)
        
        if self.qual_vars and self.quant_vars:
            # Para não explodir a dimensionalidade, usamos apenas a primeira variável qualitativa
            main_qual = self.qual_vars[0]
            for quant in self.quant_vars:
                # Criar termo de interação
                interaction_name = f"inter_{main_qual}_X_{quant}"
                # Preencher NaNs com a média antes de multiplicar
                df_a3[main_qual] = df_a3[main_qual].fillna(df_a3[main_qual].mean())
                df_a3[quant] = df_a3[quant].fillna(df_a3[quant].mean())
                df_a3[interaction_name] = df_a3[main_qual] * df_a3[quant]
                    
            print(f"[{self.dataset_name}] Criadas {len(self.quant_vars)} variáveis de interação com {main_qual}.")
            
        return df_a3

    def process_all_strategies(self):
        """Aplica as 3 estratégias e retorna um dicionário com os DataFrames."""
        # Se for o dataset não agregado, as estratégias são iguais (não há vars qualitativas)
        if self.dataset_name == 'nao_agregado':
            print(f"[{self.dataset_name}] Dataset não agregado: Estratégias A1, A2, A3 são idênticas.")
            df_base = self.df.copy()
            return {
                'A1_Direta': df_base,
                'A2_PCA': df_base,
                'A3_Interacao': df_base
            }
            
        return {
            'A1_Direta': self.apply_strategy_a1(),
            'A2_PCA': self.apply_strategy_a2(),
            'A3_Interacao': self.apply_strategy_a3()
        }

def load_and_process_datasets():
    """Carrega os datasets originais e aplica a engenharia de features."""
    results = {}
    
    for name, filename in config.DATASETS.items():
        filepath = os.path.join(config.DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ERRO CRÍTICO: Ficheiro não encontrado: {filepath}\n"
                                  f"Verifique se o ficheiro existe em {config.DATA_DIR}")
        
        print(f"\nProcessando dataset: {name} ({filename})")
        df = pd.read_csv(filepath)
        
        print(f"  Linhas originais: {len(df)}")
        print(f"  Colunas: {df.shape[1]}")
        print(f"  NaNs na coluna alvo: {df[config.TARGET_VAR].isna().sum()}")
        
        engineer = FeatureEngineer(df, name)
        strategies_dfs = engineer.process_all_strategies()
        
        # Salvar os novos datasets
        for strategy, df_strat in strategies_dfs.items():
            out_filename = f"{name}_{strategy}.csv"
            out_filepath = os.path.join(config.OUTPUT_DIR, out_filename)
            df_strat.to_csv(out_filepath, index=False)
            print(f"  -> Salvo: {out_filename} (Shape: {df_strat.shape})")
            
        results[name] = strategies_dfs
            
    return results

if __name__ == "__main__":
    load_and_process_datasets()
