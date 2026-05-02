import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import passo4_model_train_config as config

class TrainingVisualizer:
    def __init__(self):
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_treino')
        # Garantir que o diretório pai existe
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_training_history(self, model_name, dataset_name, strategy_name, history):
        """Visualiza o histórico de treinamento (loss) para modelos que suportam (ex: XGBoost, Redes Neurais)."""
        print(f"Gerando gráfico de histórico de treino para {model_name} ({dataset_name}-{strategy_name})...")
        
        plt.figure(figsize=(8, 5))
        
        if isinstance(history, dict) and 'validation_0' in history:
            # Formato XGBoost
            epochs = len(history['validation_0']['rmse'])
            x_axis = range(0, epochs)
            
            plt.plot(x_axis, history['validation_0']['rmse'], label='Train')
            if 'validation_1' in history:
                plt.plot(x_axis, history['validation_1']['rmse'], label='Validation')
                
            plt.title(f'Histórico de Treinamento - {model_name} ({dataset_name}-{strategy_name})')
            plt.xlabel('Épocas (Boosting Rounds)')
            plt.ylabel('RMSE')
            plt.legend()
            
            filename = f'history_{model_name}_{dataset_name}_{strategy_name}.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()
        else:
            print(f"  -> Histórico não disponível ou formato não suportado para {model_name}.")

    def plot_model_comparison_dummy(self):
        """Gera um gráfico comparativo dummy para ilustrar o processo de treino."""
        print("Gerando gráfico comparativo de tempo de treino (dummy)...")
        
        data = {
            'Modelo': ['RandomForest', 'XGBoost', 'SARIMAX'],
            'Tempo (s)': [12.5, 8.2, 45.1]
        }
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Modelo', y='Tempo (s)', data=df, palette='viridis')
        plt.title('Tempo de Treinamento por Modelo (Ilustrativo)')
        plt.ylabel('Segundos')
        
        filename = 'comparacao_tempo_treino.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

if __name__ == "__main__":
    viz = TrainingVisualizer()
    viz.plot_model_comparison_dummy()
