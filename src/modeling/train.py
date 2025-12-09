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
        self.features_path = PROCESSED_DIR / "features" / "player_features_v2_position.parquet"
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
            # IDs
            'player_id', 'game_id', 'team_id', 'teamid', 'team_id_opp',
            
            # Stats del partido actual (LEAKAGE)
            'min', 'plusminuspoints', 'ppm',
            'pts', 'reb', 'ast', 'stl', 'blk', 'tov',
            'dreb', 'oreb',
            'fieldgoalsmade', 'fieldgoalsattempted', 'fieldgoalspercentage',
            'threepointersmade', 'threepointersattempted', 'threepointerspercentage',
            'freethrowsmade', 'freethrowsattempted', 'freethrowspercentage',
            'foulspersonal',
            
            # Stats de equipo del partido actual (LEAKAGE de team_stats.py)
            'possessions', 'team_pace',  # Calculados con datos del partido actual
            'points_allowed', 'points_allowed_actual',
            'opp_pace_actual',  # Pace del rival en ESTE partido
        ]
        
        # Features válidas: rolling stats + contexto + PACE/DEFENSA DEL RIVAL
        features = [
            col for col in numeric_cols
            if (
                ('last' in col or 'std' in col or 'max_10' in col or 'min_10' in col)
                and col not in forbidden_cols
            )
        ]
        
        # Agregar features de contexto
        context_features = ['is_home', 'rest_days']
        for cf in context_features:
            if cf in numeric_cols and cf not in features:
                features.append(cf)
        
        # Agregar stats de equipo/rival (las que son HISTÓRICAS, no del partido actual)
        team_features = ['pace_last_10', 'opp_pace_last_10', 'opp_pts_allowed_last_10']
        for tf in team_features:
            if tf in numeric_cols and tf not in features:
                features.append(tf)
        # === NUEVAS: Features de Mercado (Fase 1) ===
        market_features = [
            'vegas_total',        # O/U del partido
            'vegas_spread',       # Spread (+ underdog, - favorito)
            'is_favorite',        # 1 si es favorito, 0 si no
            'expected_blowout',   # 1 si |spread| > 10
            'implied_team_score', # Puntos esperados del equipo
        ]
        for mf in market_features:
            if mf in numeric_cols and mf not in features:
                features.append(mf)
        leakage_check = ['min', 'possessions', 'team_pace', 'opp_pace_actual']
        for lc in leakage_check:
            if lc in features:
                raise ValueError(f"ALERTA: '{lc}' se coló. Esto es Data Leakage.")
                # === FASE 2: Features de Defensa por Posición ===
        position_features = [
            'opp_pts_to_pos_last_10',   # Puntos que el rival permite a esta posición
            'opp_def_rating_last_5',    # Rating defensivo del rival
        ]
        for pf in position_features:
            if pf in numeric_cols and pf not in features:
                features.append(pf)

        #print(f"Features seleccionadas ({len(features)}): {features}")

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
            objective='reg:absoluteerror',
            n_estimators=407, 
            learning_rate=0.038601627137871235,
            max_depth=3, 
            subsample=0.7729681200178756, 
            colsample_bytree=0.8199059157478101, 
            reg_alpha=0.045923682461824385, 
            reg_lambda=0.46299918205971335
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

        return model, X_test, y_test, predictions, features

if __name__ == "__main__":
    trainer = ModelTrainer()
    try:
        model, X_test, y_test, preds, features = trainer.train_points_model()
        
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
    
        
        # --- Visualizar Importancia de Variables ---
        # Esto nos dirá si el modelo está usando las Odds de Vegas
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=20, importance_type='weight', title='Top 20 Variables (Peso)')
        plt.tight_layout()
        plt.show()
        


        
    except Exception as e:
        logger.error(f"Ocurrió un error: {e}")