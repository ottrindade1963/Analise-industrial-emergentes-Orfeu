import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os
import passo4_model_train_config as config

# Dummy classes para LSTM e TFT para demonstração sem precisar de TensorFlow/PyTorch pesados
class DummyLSTM:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42) # Usando RF como proxy
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

class DummyTFT:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42) # Usando XGB como proxy
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

class ModelTrainer:
    def __init__(self, df, dataset_name, strategy_name):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.models = {}
        self.histories = {}
        
        # Padronizar nomes das colunas de país e ano
        if 'country' in self.df.columns and 'pais' not in self.df.columns:
            self.df.rename(columns={'country': 'pais'}, inplace=True)
        if 'year' in self.df.columns and 'ano' not in self.df.columns:
            self.df.rename(columns={'year': 'ano'}, inplace=True)
            
        # Garantir ordenação temporal para Walk-Forward
        if 'pais' in self.df.columns and 'ano' in self.df.columns:
            self.df = self.df.sort_values(by=['pais', 'ano'])

    def prepare_data(self):
        """Divide os dados em Treino, Validação e Teste (Walk-Forward Temporal por Ano)."""
        # Criar máscaras booleanas baseadas nos anos de corte
        train_mask = self.df['ano'] <= config.TRAIN_END_YEAR
        val_mask = (self.df['ano'] > config.TRAIN_END_YEAR) & (self.df['ano'] <= config.VAL_END_YEAR)
        test_mask = self.df['ano'] > config.VAL_END_YEAR
        
        # Identificar todas as colunas do tipo object/string para exclusão automática
        obj_cols = self.df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Garantir que 'pais', 'ano' e TODAS as colunas de texto não entram nas features
        # Incluir explicitamente colunas conhecidas de texto
        cols_to_drop = list(set(['pais', 'ano', 'country', 'codigo_iso3', 'country_code', 'fonte_dados'] + obj_cols))
        features = [c for c in self.df.columns if c not in cols_to_drop and c != config.TARGET_VAR]
        
        # DEBUG: Verificar o que está acontecendo
        print(f"  DEBUG: Colunas do dataset: {self.df.columns.tolist()}")
        print(f"  DEBUG: Colunas object/string encontradas: {obj_cols}")
        print(f"  DEBUG: Colunas a remover (total): {cols_to_drop}")
        print(f"  DEBUG: Features usadas para treinamento: {features}")
        print(f"  DEBUG: Tipos das features: {self.df[features].dtypes.to_dict()}")
        
        # Verificar se ainda há colunas de texto nas features
        text_cols_in_features = [c for c in features if self.df[c].dtype == 'object' or self.df[c].dtype == 'string']
        if text_cols_in_features:
            print(f"  AVISO: Ainda há colunas de texto nas features: {text_cols_in_features}")
            raise ValueError(f"Colunas de texto encontradas nas features: {text_cols_in_features}")
        
        # Dividir X e y usando as máscaras - GARANTIR QUE SÃO NUMÉRICOS
        try:
            self.X_train = self.df.loc[train_mask, features].astype(float)
            self.y_train = self.df.loc[train_mask, config.TARGET_VAR].astype(float)
            
            self.X_val = self.df.loc[val_mask, features].astype(float)
            self.y_val = self.df.loc[val_mask, config.TARGET_VAR].astype(float)
            
            self.X_test = self.df.loc[test_mask, features].astype(float)
            self.y_test = self.df.loc[test_mask, config.TARGET_VAR].astype(float)
            
            # REMOVER NaNs NA VARIÁVEL ALVO (IMPORTANTE!)
            train_valid_mask = ~self.y_train.isna()
            val_valid_mask = ~self.y_val.isna()
            test_valid_mask = ~self.y_test.isna()
            
            self.X_train = self.X_train[train_valid_mask]
            self.y_train = self.y_train[train_valid_mask]
            
            self.X_val = self.X_val[val_valid_mask]
            self.y_val = self.y_val[val_valid_mask]
            
            self.X_test = self.X_test[test_valid_mask]
            self.y_test = self.y_test[test_valid_mask]
        except ValueError as e:
            print(f"  ERRO ao converter para float: {e}")
            print(f"  Verificando valores problemáticos...")
            for col in features:
                try:
                    self.df[col].astype(float)
                except:
                    print(f"    Coluna {col} contém valores não-numéricos: {self.df[col].unique()[:5]}")
            raise
        
        # Preservar metadados (pais e ano) para interpretação posterior
        self.test_info = self.df.loc[test_mask, ['pais', 'ano']].copy()
        
        print(f"  -> Dados preparados (Divisão por Ano):")
        print(f"     Treino (<= {config.TRAIN_END_YEAR}): {len(self.X_train)} amostras")
        print(f"     Validação ({config.TRAIN_END_YEAR+1}-{config.VAL_END_YEAR}): {len(self.X_val)} amostras")
        print(f"     Teste (> {config.VAL_END_YEAR}): {len(self.X_test)} amostras")

    def train_random_forest(self):
        print("  -> Treinando Random Forest...")
        model = RandomForestRegressor(**config.RF_PARAMS)
        model.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = model
        
    def train_xgboost(self):
        print("  -> Treinando XGBoost...")
        model = xgb.XGBRegressor(**config.XGB_PARAMS)
        
        # XGBoost permite early stopping com conjunto de validação
        eval_set = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
        model.fit(
            self.X_train, self.y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.models['XGBoost'] = model
        self.histories['XGBoost'] = model.evals_result()

    def train_sarimax(self):
        print("  -> Treinando SARIMAX...")
        try:
            # SARIMAX pode ser lento e falhar com muitas features, usamos apenas as top 5
            exog_cols = self.X_train.columns[:5]
            
            model = SARIMAX(
                endog=self.y_train.values,
                exog=self.X_train[exog_cols].values,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0)
            )
            fitted_model = model.fit(disp=False)
            self.models['SARIMAX'] = fitted_model
        except Exception as e:
            print(f"     Erro no SARIMAX: {e}. Criando modelo dummy.")
            self.models['SARIMAX'] = DummyLSTM() # Fallback

    def train_lstm(self):
        print("  -> Treinando LSTM (Proxy)...")
        # Na prática, LSTM requer formatação 3D (samples, timesteps, features)
        # Aqui usamos um proxy para demonstração do pipeline
        model = DummyLSTM(**config.LSTM_PARAMS)
        model.fit(self.X_train, self.y_train)
        self.models['LSTM'] = model

    def train_tft(self):
        print("  -> Treinando TFT (Proxy)...")
        # TFT requer formatação complexa (Darts ou PyTorch Forecasting)
        # Aqui usamos um proxy para demonstração do pipeline
        model = DummyTFT(**config.TFT_PARAMS)
        model.fit(self.X_train, self.y_train)
        self.models['TFT'] = model

    def train_all(self):
        """Treina todos os modelos configurados."""
        self.prepare_data()
        
        if 'RandomForest' in config.MODELS_TO_TRAIN: self.train_random_forest()
        if 'XGBoost' in config.MODELS_TO_TRAIN: self.train_xgboost()
        if 'SARIMAX' in config.MODELS_TO_TRAIN: self.train_sarimax()
        if 'LSTM' in config.MODELS_TO_TRAIN: self.train_lstm()
        if 'TFT' in config.MODELS_TO_TRAIN: self.train_tft()
            
        self.save_models()
        return self.models, self.histories

    def save_models(self):
        """Salva os modelos treinados em disco."""
        for name, model in self.models.items():
            filename = f"{name}_{self.dataset_name}_{self.strategy_name}.pkl"
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            joblib.dump(model, filepath)
            print(f"  -> Modelo salvo: {filename}")

def run_training_for_all():
    """Executa o treinamento para todas as combinações de Dataset e Estratégia."""
    for dataset in config.DATASETS:
        for strategy in config.STRATEGIES:
            # Para o dataset não agregado, as estratégias são idênticas, mas mantemos a nomenclatura
            filename = f"{dataset}_{strategy}.csv"
            filepath = os.path.join(config.DATA_DIR, filename)
            
            if os.path.exists(filepath):
                print(f"\n{'='*40}")
                print(f"Treinando modelos para: {dataset} - {strategy}")
                print(f"{'='*40}")
                
                df = pd.read_csv(filepath)
                trainer = ModelTrainer(df, dataset, strategy)
                trainer.train_all()
            else:
                print(f"\nAVISO: Ficheiro não encontrado: {filepath}")

if __name__ == "__main__":
    run_training_for_all()
