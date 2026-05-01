import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import passo5_eval_config as config

class EvaluationVisualizer:
    def __init__(self, results_df):
        self.results_df = results_df
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_avaliacao')
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_rmse_comparison(self):
        """Gera gráfico de barras comparando o RMSE dos modelos por estratégia."""
        print("Gerando gráfico de comparação de RMSE...")
        
        plt.figure(figsize=(14, 8))
        
        # Agrupar por Modelo e Estratégia (média dos datasets)
        avg_rmse = self.results_df.groupby(['Modelo', 'Estrategia'])['RMSE'].mean().reset_index()
        
        sns.barplot(x='Modelo', y='RMSE', hue='Estrategia', data=avg_rmse, palette='viridis')
        
        plt.title('Comparação de RMSE por Modelo e Estratégia de Agregação (Média dos Datasets)', fontsize=16)
        plt.ylabel('RMSE (Menor é Melhor)')
        plt.xlabel('Modelo Preditivo')
        plt.legend(title='Estratégia', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = 'comparacao_rmse_modelos_estrategias.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_performance_heatmap(self):
        """Gera um mapa de calor com as métricas de performance."""
        print("Gerando heatmap de performance...")
        
        # Pivotar a tabela para ter Modelos nas linhas e Estratégias nas colunas
        pivot_rmse = self.results_df.pivot_table(
            index='Modelo', 
            columns='Estrategia', 
            values='RMSE', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_rmse, annot=True, cmap='YlGnBu_r', fmt='.3f', linewidths=.5)
        
        plt.title('Heatmap de RMSE (Média dos Datasets)', fontsize=16)
        plt.ylabel('Modelo')
        plt.xlabel('Estratégia de Agregação')
        
        plt.tight_layout()
        filename = 'heatmap_rmse_performance.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_dataset_comparison(self):
        """Compara a performance entre os diferentes datasets (Inner, Left, Outer)."""
        print("Gerando gráfico de comparação de datasets...")
        
        plt.figure(figsize=(12, 6))
        
        # Agrupar por Dataset e Modelo
        avg_rmse = self.results_df.groupby(['Dataset', 'Modelo'])['RMSE'].mean().reset_index()
        
        sns.barplot(x='Dataset', y='RMSE', hue='Modelo', data=avg_rmse, palette='Set2')
        
        plt.title('Comparação de RMSE por Dataset e Modelo (Média das Estratégias)', fontsize=16)
        plt.ylabel('RMSE (Menor é Melhor)')
        plt.xlabel('Dataset Agregado')
        
        plt.tight_layout()
        filename = 'comparacao_rmse_datasets.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações de avaliação."""
        if self.results_df is not None and not self.results_df.empty:
            self.plot_rmse_comparison()
            self.plot_performance_heatmap()
            self.plot_dataset_comparison()
            print(f"Todas as visualizações salvas em: {self.output_dir}")
        else:
            print("Nenhum dado de resultado disponível para visualização.")

if __name__ == "__main__":
    # Teste simples
    import passo5_eval_processor as processor
    evaluator = processor.ModelEvaluator()
    results_df = evaluator.run_evaluation()
    viz = EvaluationVisualizer(results_df)
    viz.generate_all_visualizations()
