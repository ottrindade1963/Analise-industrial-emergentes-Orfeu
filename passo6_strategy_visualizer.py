import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import passo6_strategy_config as config

class StrategyVisualizer:
    def __init__(self, analysis_df_datasets, analysis_df_strategies):
        self.analysis_df_datasets = analysis_df_datasets
        self.analysis_df_strategies = analysis_df_strategies
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_estrategias')
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_gain_vs_non_aggregated(self):
        """Gera gráfico de barras mostrando o ganho de usar dados agregados vs não agregados."""
        if self.analysis_df_datasets is None:
            return
            
        print("Gerando gráfico de ganho: Agregados vs Não Agregados...")
        
        # Agrupar por Modelo e Dataset (média das estratégias)
        avg_gain = self.analysis_df_datasets.groupby(['Modelo', 'Dataset'])['Melhoria_RMSE_%'].mean().reset_index()
        
        plt.figure(figsize=(12, 7))
        
        palette = sns.color_palette("Set2", n_colors=len(avg_gain['Dataset'].unique()))
        
        ax = sns.barplot(x='Modelo', y='Melhoria_RMSE_%', hue='Dataset', data=avg_gain, palette=palette)
        
        plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
        
        plt.title('Melhoria Percentual do RMSE ao usar Dados Agregados (vs Não Agregados)', fontsize=16)
        plt.ylabel('Melhoria no RMSE (%) - Valores Positivos são Melhores')
        plt.xlabel('Modelo Preditivo')
        
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.annotate(f'{height:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom' if height > 0 else 'top', 
                            xytext=(0, 5 if height > 0 else -15), 
                            textcoords='offset points')
                            
        plt.tight_layout()
        filename = 'ganho_agregados_vs_nao_agregados.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_gain_between_strategies(self):
        """Gera gráfico de barras mostrando o ganho de A2 e A3 em relação a A1."""
        if self.analysis_df_strategies is None:
            return
            
        print("Gerando gráfico de ganho entre estratégias (A2/A3 vs A1)...")
        
        # Filtrar apenas A2 e A3 (A1 tem ganho 0% por definição)
        df_plot = self.analysis_df_strategies[self.analysis_df_strategies['Estrategia'] != config.BASELINE_STRATEGY].copy()
        
        # Agrupar por Estratégia e Modelo (média dos datasets)
        avg_gain = df_plot.groupby(['Estrategia', 'Modelo'])['Melhoria_RMSE_vs_A1_%'].mean().reset_index()
        
        plt.figure(figsize=(12, 7))
        
        palette = sns.color_palette("coolwarm", n_colors=len(avg_gain['Estrategia'].unique()))
        
        ax = sns.barplot(x='Modelo', y='Melhoria_RMSE_vs_A1_%', hue='Estrategia', data=avg_gain, palette=palette)
        
        plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
        
        plt.title('Melhoria Percentual do RMSE em Relação à Estratégia A1 (Inclusão Direta)', fontsize=16)
        plt.ylabel('Melhoria no RMSE (%) - Valores Positivos são Melhores')
        plt.xlabel('Modelo Preditivo')
        
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.annotate(f'{height:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom' if height > 0 else 'top', 
                            xytext=(0, 5 if height > 0 else -15), 
                            textcoords='offset points')
                            
        plt.tight_layout()
        filename = 'ganho_percentual_estrategias_a2_a3.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações de análise de estratégias."""
        if self.analysis_df_datasets is not None and not self.analysis_df_datasets.empty:
            self.plot_gain_vs_non_aggregated()
            self.plot_gain_between_strategies()
            print(f"Todas as visualizações salvas em: {self.output_dir}")
        else:
            print("Nenhum dado de análise disponível para visualização.")

if __name__ == "__main__":
    # Teste simples
    import passo6_strategy_processor as processor
    analyzer = processor.StrategyAnalyzer()
    df_datasets, df_strategies = analyzer.run_analysis()
    viz = StrategyVisualizer(df_datasets, df_strategies)
    viz.generate_all_visualizations()
