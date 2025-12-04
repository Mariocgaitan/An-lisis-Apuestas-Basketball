import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from src.config import PROCESSED_DIR

def objective(trial):
    # 1. Cargar Datos
    df = pd.read_parquet(PROCESSED_DIR / "features" / "player_features_v2.parquet") # O v2
    df = df.dropna(subset=['pts_last_5', 'pts'])
    df = df.sort_values('game_date')
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

    X = df[features]
    y = df['pts']
    
    # Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 2. El Espacio de Búsqueda (Hyperparameter Space)
    param = {
        'objective': 'reg:absoluteerror', # Optimizar MAE directamente
        'tree_method': 'hist', # Más rápido
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # L1 Reg
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2 Reg
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100)
    }
    
    # 3. Entrenar
    model = xgb.XGBRegressor(**param, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 4. Evaluar
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    return mae

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # Probará 50 combinaciones
    
    print("Mejores parámetros:", study.best_params)
    print("Mejor MAE:", study.best_value)