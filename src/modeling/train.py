import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from src.config import PROCESSED_DIR, BASE_DIR, LOGS_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.features_path = PROCESSED_DIR / "features" / "player_features_v2.parquet"
        self.model_path = BASE_DIR / "src" / "modeling" / "model_store"
        self.model_path.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Carga los datos y elimina filas sin historia suficiente"""
        if not self.features_path.exists():
            raise FileNotFoundError("No se encontró player_features.parquet. Corre features.py primero.")
        
        df = pd.read_parquet(self.features_path)
        
        # Eliminamos filas donde pts_last_5 sea NaN
        df_clean = df.dropna(subset=['pts_last_5', 'pts'])
        
        # Ordenamos cronológicamente
        df_clean = df_clean.sort_values(by='game_date')
        
        return df_clean

    def train_points_model(self):
        df = self.load_data()
        
        target = 'pts'
        
        # --- CORRECCIÓN AQUÍ ---
        # 1. Seleccionamos solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
        # Columnas PROHIBIDAS (Data Leakage)
        forbidden_cols = [
            'player_id', 'game_id', 'team_id', 'teamid',
            'min', 'plusminuspoints', 'ppm',
            'pts', 'reb', 'ast', 'stl', 'blk', 'tov',
            'dreb', 'oreb',
            'fieldgoalsmade', 'fieldgoalsattempted',
            'threepointersmade', 'threepointersattempted',
            'freethrowsmade', 'freethrowsattempted',
        ]
        features = [
            col for col in numeric_cols
            if (('last' in col or 'std' in col or 'rest' in col or 'is_home' in col or 'ppm' in col or 'min' in col or 'max' in col or 'min' in col) and col not in forbidden_cols)
            ]   
        
        context_features = ['is_home', 'rest_days']
        for cf in context_features:
            if cf in numeric_cols and cf not in features:
                features.append(cf)
        
        # Validación de seguridad
        if 'min' in features:
            raise ValueError("ALERTA: 'min' se coló. Esto es Data Leakage.")
        logger.info(f"Entrenando con {len(df)} registros.")
        logger.info(f"Variables seleccionadas ({len(features)}): {features}")

        X = df[features]
        y = df['pts']
        

        # Time Series Split (80/20)
        split_idx = int(len(df) * 0.8)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

        # Modelo XGBoost
                # Modelo XGBoost con parámetros optimizados por Optuna
        model = xgb.XGBRegressor(
            n_estimators=156,
            learning_rate=0.0737887350250719,
            max_depth=3,
            subsample=0.9328794723122931,
            colsample_bytree=0.6510860646397484,
            reg_alpha=0.03689658798503173,
            reg_lambda=0.005018567629789179,
            early_stopping_rounds=19,
            random_state=42
        )

        logger.info("Iniciando entrenamiento...")
        # XGBoost requiere validation set para el early_stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )

        # Evaluación
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        logger.info(f"=== RESULTADOS ===")
        logger.info(f"MAE (Error Promedio Absoluto): {mae:.2f} puntos")
        
        # Guardar modelo
        save_loc = self.model_path / "xgb_points_v1.pkl"
        joblib.dump(model, save_loc)
        logger.info(f"Modelo guardado en: {save_loc}")

        return model, X_test, y_test, predictions

if __name__ == "__main__":
    trainer = ModelTrainer()
    try:
        model, X_test, y_test, preds = trainer.train_points_model()
        
        # Visualizar predicción vs realidad
        comparison = pd.DataFrame({
            'Realidad': y_test.values, 
            'Predicción': preds
        })
        comparison['Error'] = comparison['Realidad'] - comparison['Predicción']
        
        print("\n--- Predicción vs Realidad (Muestra de 10) ---")
        print(comparison.head(10))
        
        print("\n--- Estadísticas de Error ---")
        print(comparison['Error'].describe())
        
    except Exception as e:
        logger.error(f"Ocurrió un error: {e}")