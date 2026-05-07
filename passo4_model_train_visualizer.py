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

    def plot_real_training_metrics(self):
        """Gera gráficos reais baseados nos logs de treinamento salvos."""
        print("Gerando gráficos reais de treinamento...")
        
        log_path = os.path.join(config.OUTPUT_DIR, 'training_logs.pkl')
        if not os.path.exists(log_path):
            print(f"  -> Arquivo de logs não encontrado: {log_path}")
            return
            
        import joblib
        logs = joblib.load(log_path)
        
        if not logs:
            print("  -> Logs de treinamento vazios.")
            return
            
        # Extrair dados para DataFrame
        records = []
        for key, data in logs.items():
            # key formato: "Modelo_Dataset_Estrategia"
            parts = key.split('_', 1)
            if len(parts) == 2:
                model = parts[0]
                dataset_strat = parts[1]
                
                # Separar dataset e estrategia para análises mais ricas
                ds_parts = dataset_strat.split('_A')
                if len(ds_parts) == 2:
                    dataset = ds_parts[0]
                    strategy = 'A' + ds_parts[1]
                else:
                    dataset = dataset_strat
                    strategy = 'N/A'
                
                records.append({
                    'Modelo': model,
                    'Cenario': dataset_strat,
                    'Dataset': dataset,
                    'Estrategia': strategy,
                    'RMSE_Treino': data.get('train_rmse', 0),
                    'R2_Treino': data.get('train_r2', 0),
                    'Tempo_s': data.get('train_time', 0)
                })
                
        if not records:
            return
            
        df = pd.DataFrame(records)
        
        # 1. Gráfico de Tempo de Treinamento Real
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Modelo', y='Tempo_s', hue='Cenario', data=df)
        plt.title('Tempo de Treinamento Real por Modelo e Cenário', fontsize=14)
        plt.ylabel('Tempo (segundos)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tempo_treino_real.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gráfico de R2 no Treino
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Modelo', y='R2_Treino', hue='Cenario', data=df)
        plt.title('R² no Conjunto de Treinamento', fontsize=14)
        plt.ylabel('R² Score')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'r2_treino_real.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Gráfico de RMSE no Treino
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Modelo', y='RMSE_Treino', hue='Cenario', data=df)
        plt.title('RMSE no Conjunto de Treinamento', fontsize=14)
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rmse_treino_real.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap de R2 por Modelo e Cenário
        plt.figure(figsize=(14, 8))
        pivot_r2 = df.pivot(index='Modelo', columns='Cenario', values='R2_Treino')
        sns.heatmap(pivot_r2, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
        plt.title('Heatmap de R² no Treinamento por Modelo e Cenário', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'heatmap_r2_treino.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Boxplot de RMSE por Modelo
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Modelo', y='RMSE_Treino', data=df, palette='Set3')
        plt.title('Distribuição do RMSE de Treinamento por Modelo', fontsize=14)
        plt.ylabel('RMSE')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'boxplot_rmse_treino.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Comparação de Datasets (Média de RMSE)
        plt.figure(figsize=(10, 6))
        avg_rmse_ds = df.groupby('Dataset')['RMSE_Treino'].mean().reset_index()
        sns.barplot(x='Dataset', y='RMSE_Treino', data=avg_rmse_ds, palette='coolwarm')
        plt.title('RMSE Médio de Treinamento por Dataset', fontsize=14)
        plt.ylabel('RMSE Médio')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rmse_medio_por_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  -> 6 Gráficos reais gerados em {self.output_dir}")

if __name__ == "__main__":
    viz = TrainingVisualizer()
    viz.plot_real_training_metrics()
