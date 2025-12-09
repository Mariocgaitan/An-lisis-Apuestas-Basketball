import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from src.config import PROCESSED_DIR

def objective(trial):
    # 1. Cargar Datos
    features_path = PROCESSED_DIR / "features" / "player_features_v2_position.parquet"
    
    
    df = pd.read_parquet(features_path)
    df = df.dropna(subset=['pts_last_5', 'pts'])
    df = df.sort_values('game_date').reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Columnas PROHIBIDAS (Data Leakage)
    forbidden_cols = [
        'player_id', 'game_id', 'team_id', 'teamid', 'team_id_opp',
        'min', 'plusminuspoints', 'ppm',
        'pts', 'reb', 'ast', 'stl', 'blk', 'tov',
        'dreb', 'oreb',
        'fieldgoalsmade', 'fieldgoalsattempted', 'fieldgoalspercentage',
        'threepointersmade', 'threepointersattempted', 'threepointerspercentage',
        'freethrowsmade', 'freethrowsattempted', 'freethrowspercentage',
        'foulspersonal',
        'possessions', 'team_pace',
        'points_allowed', 'points_allowed_actual',
        'opp_pace_actual',
    ]
    
    features = [
        col for col in numeric_cols
        if (('last' in col or 'std' in col or 'max_10' in col or 'min_10' in col)
            and col not in forbidden_cols)
    ]
    
    context_features = ['is_home', 'rest_days']
    for cf in context_features:
        if cf in numeric_cols and cf not in features:
            features.append(cf)
    
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

    X = df[features]
    y = df['pts']
    
    # === VALIDACIÓN CRUZADA TEMPORAL (Time Series Split) ===
    # En lugar de un solo split, hacemos 3 splits temporales
    n = len(df)
    fold_maes = []
    
    # Fold 1: Train 0-50%, Test 50-60%
    # Fold 2: Train 0-60%, Test 60-70%
    # Fold 3: Train 0-70%, Test 70-80%
    folds = [
        (int(n * 0.50), int(n * 0.65)),  # Train: 0-50%, Test: 50-65%
        (int(n * 0.65), int(n * 0.80)),  # Train: 0-65%, Test: 65-80%
        (int(n * 0.80), int(n * 1.00)),  # Train: 0-80%, Test: 80-100% (igual que train.py)
    ]
    
    # Hiperparámetros a probar
    param = {
        'objective': 'reg:absoluteerror',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
        'early_stopping_rounds': 50,  # Fijo, no optimizar
        'random_state': 42
    }
    
    for train_end, test_end in folds:
        X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
        y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        fold_maes.append(mae)
    
    # Retornamos el PROMEDIO de los 3 folds
    return np.mean(fold_maes)

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print("\n" + "="*50)
    print("Mejores parámetros:", study.best_params)
    print("Mejor MAE (promedio 3 folds):", study.best_value)