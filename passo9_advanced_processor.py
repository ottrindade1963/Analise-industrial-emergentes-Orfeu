import pandas as pd
import numpy as np
import joblib
import os
import passo9_advanced_config as config

class AdvancedAnalyzer:
    def __init__(self):
        self.sensitivity_results = []
        self.robustness_results = []

    def load_data_and_model(self, model_name, dataset_name, strategy_name):
        """Carrega os dados de teste e o modelo treinado."""
        data_filename = f"{dataset_name}_{strategy_name}.csv"
        data_filepath = os.path.join(config.DATA_DIR, data_filename)
        
        if not os.path.exists(data_filepath):
            return None, None
            
        df = pd.read_csv(data_filepath)
        
        country_col = 'pais' if 'pais' in df.columns else ('country' if 'country' in df.columns else None)
        year_col = 'ano' if 'ano' in df.columns else ('year' if 'year' in df.columns else None)
        
        if country_col and year_col:
            df = df.sort_values(by=[country_col, year_col])
            
        cols_to_drop = ['country', 'year', 'ano', 'pais']
        features = [c for c in df.columns if c not in cols_to_drop and c != config.TARGET_VAR]
        
        X = df[features]
        
        # Identificar a coluna de ano
        year_col = 'year' if 'year' in df.columns else 'ano'
        
        # Importar config do passo 4 para ter os anos de corte
        import passo4_model_train_config as train_config
        
        # Máscara para o conjunto de teste
        test_mask = df[year_col] > train_config.VAL_END_YEAR
        
        X_test = X.loc[test_mask]
        
        model_filename = f"{model_name}_{dataset_name}_{strategy_name}.pkl"
        model_filepath = os.path.join(config.MODEL_DIR, model_filename)
        
        if not os.path.exists(model_filepath):
            return None, None
            
        model = joblib.load(model_filepath)
        
        return model, X_test

    def run_sensitivity_analysis(self, model, X_test, model_name):
        """Analisa o impacto de variações nas variáveis-chave na previsão final."""
        print(f"  -> Executando Análise de Sensibilidade para {model_name}...")
        
        # Previsão base (sem alterações)
        base_preds = model.predict(X_test)
        base_mean = np.mean(base_preds)
        
        for var in config.SENSITIVITY_VARS:
            if var in X_test.columns:
                for step in config.SENSITIVITY_STEPS:
                    if step == 0.0:
                        continue
                        
                    # Criar cópia e aplicar variação
                    X_modified = X_test.copy()
                    
                    # Se a variável for padronizada (média 0), a variação percentual não funciona bem
                    # Usamos desvio padrão como base de variação
                    std_dev = X_modified[var].std()
                    X_modified[var] = X_modified[var] + (std_dev * step * 5) # Multiplicador para efeito visível
                    
                    # Nova previsão
                    new_preds = model.predict(X_modified)
                    new_mean = np.mean(new_preds)
                    
                    # Calcular impacto percentual na previsão média
                    impact_pct = ((new_mean - base_mean) / base_mean) * 100
                    
                    self.sensitivity_results.append({
                        'Modelo': model_name,
                        'Variavel': var,
                        'Variacao_Aplicada': f"{step*100:+.0f}%",
                        'Impacto_Previsao_%': impact_pct
                    })

    def run_robustness_check(self, model, X_test, model_name):
        """Verifica a robustez do modelo adicionando ruído aos dados."""
        print(f"  -> Executando Teste de Robustez (Ruído) para {model_name}...")
        
        base_preds = model.predict(X_test)
        
        noise_levels = [0.01, 0.05, 0.10, 0.20] # 1%, 5%, 10%, 20% de ruído
        
        for noise in noise_levels:
            X_noisy = X_test.copy()
            
            # Adicionar ruído gaussiano a todas as features numéricas
            for col in X_noisy.columns:
                std_dev = X_noisy[col].std()
                if std_dev > 0:
                    noise_array = np.random.normal(0, std_dev * noise, size=len(X_noisy))
                    X_noisy[col] = X_noisy[col] + noise_array
                    
            noisy_preds = model.predict(X_noisy)
            
            # Calcular degradação (MAE entre previsão base e previsão com ruído)
            from sklearn.metrics import mean_absolute_error
            degradation = mean_absolute_error(base_preds, noisy_preds)
            
            self.robustness_results.append({
                'Modelo': model_name,
                'Nivel_Ruido': f"{noise*100:.0f}%",
                'Degradacao_MAE': degradation
            })

    def run_all_analyses(self):
        """Executa todas as análises avançadas."""
        print(f"\nFocando no melhor cenário: {config.BEST_DATASET} - {config.BEST_STRATEGY}")
        
        for model_name in config.BEST_MODELS:
            model, X_test = self.load_data_and_model(model_name, config.BEST_DATASET, config.BEST_STRATEGY)
            
            if model is not None and X_test is not None:
                self.run_sensitivity_analysis(model, X_test, model_name)
                self.run_robustness_check(model, X_test, model_name)
            else:
                print(f"  -> Modelo ou dados não encontrados para {model_name}")
                
        # Salvar resultados
        if self.sensitivity_results:
            df_sens = pd.DataFrame(self.sensitivity_results)
            out_sens = os.path.join(config.OUTPUT_DIR, 'analise_sensibilidade.csv')
            df_sens.to_csv(out_sens, index=False)
            
        if self.robustness_results:
            df_rob = pd.DataFrame(self.robustness_results)
            out_rob = os.path.join(config.OUTPUT_DIR, 'teste_robustez.csv')
            df_rob.to_csv(out_rob, index=False)
            
        return self.sensitivity_results, self.robustness_results

if __name__ == "__main__":
    analyzer = AdvancedAnalyzer()
    analyzer.run_all_analyses()
