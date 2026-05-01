import matplotlib.pyplot as plt
import shap
import os
import passo7_shap_config as config

class ShapVisualizer:
    def __init__(self, shap_values_dict, X_test_dict, feature_names_dict):
        self.shap_values_dict = shap_values_dict
        self.X_test_dict = X_test_dict
        self.feature_names_dict = feature_names_dict
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_shap')
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_summary(self, key):
        """Gera o gráfico de resumo SHAP (Summary Plot)."""
        print(f"Gerando SHAP Summary Plot para {key}...")
        
        shap_values = self.shap_values_dict[key]
        X_test = self.X_test_dict[key]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, max_display=config.TOP_N_FEATURES, show=False)
        
        plt.title(f'SHAP Summary Plot - {key}', fontsize=16)
        plt.tight_layout()
        
        filename = f'shap_summary_{key}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bar(self, key):
        """Gera o gráfico de barras SHAP (Importância Média Absoluta)."""
        print(f"Gerando SHAP Bar Plot para {key}...")
        
        shap_values = self.shap_values_dict[key]
        X_test = self.X_test_dict[key]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=config.TOP_N_FEATURES, show=False)
        
        plt.title(f'SHAP Feature Importance (Média Absoluta) - {key}', fontsize=16)
        plt.tight_layout()
        
        filename = f'shap_bar_{key}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_dependence_dummy(self, key):
        """Gera um gráfico de dependência SHAP dummy para demonstração."""
        print(f"Gerando SHAP Dependence Plot (dummy) para {key}...")
        
        # Para evitar erros se a feature não existir, criamos um gráfico ilustrativo
        import numpy as np
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        
        # Dados simulados
        x = np.random.uniform(-2, 2, 100)
        y = 2 * x + np.random.normal(0, 0.5, 100)
        c = np.random.uniform(0, 100, 100)
        
        scatter = plt.scatter(x, y, c=c, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Feature Interação (ex: PIB per capita)')
        
        plt.title(f'SHAP Dependence Plot (Ilustrativo) - {key}', fontsize=14)
        plt.xlabel('Valor da Feature (ex: Qualidade Regulatória)')
        plt.ylabel('Valor SHAP (Impacto na Previsão)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        filename = f'shap_dependence_dummy_{key}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações SHAP para os modelos analisados."""
        if not self.shap_values_dict:
            print("Nenhum valor SHAP disponível para visualização.")
            return
            
        for key in self.shap_values_dict.keys():
            self.plot_summary(key)
            self.plot_bar(key)
            self.plot_dependence_dummy(key)
            
        print(f"Todas as visualizações SHAP salvas em: {self.output_dir}")

if __name__ == "__main__":
    # Teste simples
    import passo7_shap_processor as processor
    analyzer = processor.ShapAnalyzer()
    shap_vals, X_test, feat_names = analyzer.run_analysis()
    viz = ShapVisualizer(shap_vals, X_test, feat_names)
    viz.generate_all_visualizations()
