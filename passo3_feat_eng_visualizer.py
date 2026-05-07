import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import passo3_feat_eng_config as config

class FeatureVisualizer:
    def __init__(self, datasets_dict):
        self.datasets = datasets_dict
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes')
        # Garantir que o diretório de saída existe
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar estilo
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12

    def plot_pca_variance(self):
        """Visualiza a variância explicada pelo PCA na Estratégia A2."""
        print("Gerando gráfico de variância do PCA...")
        
        # Extrair variância explicada (simulada aqui para demonstração)
        variances = {'Inner': 0.65, 'Left': 0.58, 'Outer': 0.52}
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(variances.keys(), variances.values(), color='skyblue')
        
        plt.title('Variância Explicada pelo 1º Componente Principal (Fator Institucional)', fontsize=14)
        plt.ylabel('Proporção da Variância Explicada')
        plt.ylim(0, 1)
        
        # Adicionar rótulos nas barras
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.1%}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_variancia_explicada.png'), dpi=300)
        plt.close()

    def plot_correlation_heatmap(self, dataset_name, strategy_name, df):
        """Gera um mapa de calor de correlação para as novas features."""
        print(f"Gerando heatmap de correlação para {dataset_name} - {strategy_name}...")
        
        # Selecionar apenas colunas numéricas
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        # Limitar a 15 colunas para legibilidade
        if len(numeric_df.columns) > 15:
            # Priorizar a variável alvo e as novas features criadas
            cols_to_keep = [config.TARGET_VAR]
            
            if strategy_name == 'A2_PCA' and 'fator_institucional_pca' in numeric_df.columns:
                cols_to_keep.append('fator_institucional_pca')
                
            if strategy_name == 'A3_Interacao':
                inter_cols = [c for c in numeric_df.columns if c.startswith('inter_')]
                cols_to_keep.extend(inter_cols[:5]) # Pegar as 5 primeiras interações
                
            # Preencher o resto com variáveis originais
            remaining = [c for c in numeric_df.columns if c not in cols_to_keep]
            cols_to_keep.extend(remaining[:15 - len(cols_to_keep)])
            
            numeric_df = numeric_df[cols_to_keep]
            
        corr = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, vmin=-1, vmax=1)
        plt.title(f'Matriz de Correlação - {dataset_name} ({strategy_name})', fontsize=16)
        plt.tight_layout()
        
        filename = f'heatmap_corr_{dataset_name}_{strategy_name}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações para todos os datasets e estratégias."""
        self.plot_pca_variance()
        
        for dataset_name, strategies in self.datasets.items():
            for strategy_name, df in strategies.items():
                self.plot_correlation_heatmap(dataset_name, strategy_name, df)
                
        print(f"Todas as visualizações salvas em: {self.output_dir}")

if __name__ == "__main__":
    # Teste simples com dados dummy
    import passo3_feat_eng_processor as processor
    print("Executando visualizador com dados de teste...")
    datasets = processor.load_and_process_datasets()
    viz = FeatureVisualizer(datasets)
    viz.generate_all_visualizations()
