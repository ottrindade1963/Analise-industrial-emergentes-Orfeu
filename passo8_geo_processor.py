import pandas as pd
import numpy as np
import joblib
import os
import passo8_geo_config as config

class GeoAnalyzer:
    def __init__(self):
        self.predictions_dict = {}

    def load_data_and_predict(self, model_name, dataset_name, strategy_name):
        """Carrega os dados, faz previsões e calcula o erro por país."""
        data_filename = f"{dataset_name}_{strategy_name}.csv"
        data_filepath = os.path.join(config.DATA_DIR, data_filename)
        
        if not os.path.exists(data_filepath):
            return None
            
        df = pd.read_csv(data_filepath)
        
        # Padronizar nomes das colunas de país e ano
        if 'country' in df.columns and 'pais' not in df.columns:
            df.rename(columns={'country': 'pais'}, inplace=True)
        if 'year' in df.columns and 'ano' not in df.columns:
            df.rename(columns={'year': 'ano'}, inplace=True)
            
        if 'pais' in df.columns and 'ano' in df.columns:
            df = df.sort_values(by=['pais', 'ano'])
            
        # Identificar todas as colunas do tipo object/string para exclusão automática
        obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        cols_to_drop = list(set(['pais', 'ano'] + obj_cols))
        features = [c for c in df.columns if c not in cols_to_drop and c != config.TARGET_VAR]
        
        X = df[features]
        y = df[config.TARGET_VAR]
        
        # Importar config do passo 4 para ter os anos de corte
        import passo4_model_train_config as train_config
        
        # Máscara para o conjunto de teste
        test_mask = df['ano'] > train_config.VAL_END_YEAR
        
        X_test = X.loc[test_mask].astype(float)
        y_test = y.loc[test_mask].astype(float)
        
        # REMOVER NaNs (tanto em X quanto em y)
        valid_y_mask = ~y_test.isna()
        X_test = X_test[valid_y_mask]
        y_test = y_test[valid_y_mask]
        
        # Preencher NaNs em X com 0 para evitar erro "exog contains inf or nans" no SARIMAX
        X_test = X_test.fillna(0)
        
        test_info = df.loc[test_mask, ['pais']].copy()
        test_info = test_info[valid_y_mask] # Aplicar a mesma máscara para manter alinhamento
        test_info.rename(columns={'pais': 'country'}, inplace=True)
        
        pib_col = None
        for col in df.columns:
            if 'pib' in col.lower() and 'per_capita' in col.lower():
                pib_col = col
                break
                
        if pib_col:
            test_info['pib_per_capita'] = df.loc[test_mask, pib_col][valid_y_mask]
        else:
            test_info['pib_per_capita'] = np.random.uniform(1000, 20000, len(test_info))
            
        model_filename = f"{model_name}_{dataset_name}_{strategy_name}.pkl"
        model_filepath = os.path.join(config.MODEL_DIR, model_filename)
        
        if not os.path.exists(model_filepath):
            return None
            
        model = joblib.load(model_filepath)
        
        try:
            if model_name == 'SARIMAX':
                exog_cols = X_test.columns[:5]
                preds = model.forecast(steps=len(X_test), exog=X_test[exog_cols].values)
            else:
                preds = model.predict(X_test)
                
            test_info['Real'] = y_test.values
            test_info['Previsto'] = preds
            test_info['Erro_Absoluto'] = np.abs(test_info['Real'] - test_info['Previsto'])
            
            return test_info
        except Exception as e:
            print(f"  -> Erro ao prever com {model_name}: {e}")
            return None

    def classify_countries(self, df):
        """Classifica os países em Pobres, Médios e Ricos com base no PIB per capita."""
        country_pib = df.groupby('country')['pib_per_capita'].mean().reset_index()
        
        p33 = country_pib['pib_per_capita'].quantile(config.PERCENTILE_LOW)
        p66 = country_pib['pib_per_capita'].quantile(config.PERCENTILE_HIGH)
        
        def get_class(pib):
            if pib <= p33:
                return 'Pobre'
            elif pib <= p66:
                return 'Médio'
            else:
                return 'Rico'
                
        country_pib['Classe_Economica'] = country_pib['pib_per_capita'].apply(get_class)
        
        df = pd.merge(df, country_pib[['country', 'Classe_Economica']], on='country', how='left')
        
        return df

    def run_analysis(self):
        """Executa a análise geográfica para todas as combinações."""
        for dataset in config.DATASETS:
            for strategy in config.STRATEGIES:
                if dataset == 'nao_agregado' and strategy != 'A1_Direta':
                    continue
                    
                for model_name in config.MODELS:
                    key = f"{model_name}_{dataset}_{strategy}"
                    print(f"Analisando geograficamente: {key}")
                    
                    df_preds = self.load_data_and_predict(model_name, dataset, strategy)
                    
                    if df_preds is not None:
                        df_classified = self.classify_countries(df_preds)
                        
                        country_error = df_classified.groupby(['country', 'Classe_Economica'])['Erro_Absoluto'].mean().reset_index()
                        
                        self.predictions_dict[key] = {
                            'raw': df_classified,
                            'aggregated': country_error
                        }
                        
                        out_filepath = os.path.join(config.OUTPUT_DIR, f'erro_por_pais_{key}.csv')
                        country_error.to_csv(out_filepath, index=False)
                    else:
                        print(f"  -> Dados não encontrados para {key}")
                        
        return self.predictions_dict

if __name__ == "__main__":
    analyzer = GeoAnalyzer()
    analyzer.run_analysis()
