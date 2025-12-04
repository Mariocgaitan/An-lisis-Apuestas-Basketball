import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DIR, PROCESSED_DIR, LOGS_DIR

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.input_dir = RAW_DIR / "nba_stats"
        self.output_dir = PROCESSED_DIR / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def standardize_v3_columns(self, df):
        """
        Traduce las columnas de la nueva API V3 a los nombres estándar (V2)
        que usamos en data science (pts, reb, ast).
        """
        # 1. Convertimos todo a minúsculas primero para asegurar coincidencias
        df.columns = [c.lower() for c in df.columns]

        # 2. Tu diccionario de mapeo (V3 -> Standard)
        rename_map = {
            'personid': 'player_id',
            'playername': 'player_name',
            'firstname': 'first_name',
            'familyname': 'last_name',
            'points': 'pts',
            'reboundsdefensive': 'dreb',
            'reboundsoffensive': 'oreb',
            'reboundstotal': 'reb',
            'assists': 'ast',
            'steals': 'stl',
            'blocks': 'blk',
            'turnovers': 'tov',
            'minutes': 'min',
            'game_date': 'game_date', 
            'matchup': 'matchup', 
            'wl': 'wl'
        }
        
        # 3. Aplicamos el cambio
        df = df.rename(columns=rename_map)
        
        # Verificación rápida
        if 'pts' not in df.columns and 'points' in df.columns:
            logger.warning("Parece que el renombrado de columnas falló. Revisa los nombres crudos.")
            
        return df

    def load_all_stats(self):
        """Carga y une todos los archivos parquet de stats diarios"""
        files = list(self.input_dir.glob("*.parquet"))
        if not files:
            logger.error("No hay datos de NBA en data/raw/nba_stats/")
            return pd.DataFrame()
        
        logger.info(f"Cargando {len(files)} archivos de estadísticas...")
        df_list = [pd.read_parquet(f) for f in files]
        full_df = pd.concat(df_list, ignore_index=True)
        
        # Estandarizamos columnas
        full_df = self.standardize_v3_columns(full_df)
        
        # Asegurarnos que la fecha sea datetime
        if 'game_date' in full_df.columns:
            full_df['game_date'] = pd.to_datetime(full_df['game_date'])
            
        return full_df

    def add_context_features(self, df):
        """Agrega contexto: Local/Visita y Descanso"""
        logger.info("Agregando contexto (Home/Away, Descanso)...")
        
        # 1. Home vs Away
        if 'matchup' in df.columns:
            # Busca ' vs. ' o ' vs ' para saber si es local
            df['is_home'] = df['matchup'].apply(lambda x: 1 if ' vs. ' in str(x) or ' vs ' in str(x) else 0)
        else:
            df['is_home'] = 0.5 

        # 2. Días de Descanso (Rest Days)
        df = df.sort_values(['player_id', 'game_date'])
        df['prev_game_date'] = df.groupby('player_id')['game_date'].shift(1)
        
        # Calculamos la diferencia en días
        df['rest_days'] = (df['game_date'] - df['prev_game_date']).dt.days
        # Llenar NAs con 3 (descanso promedio) y limitar a 7 días max para evitar outliers extremos
        df['rest_days'] = df['rest_days'].fillna(3).clip(upper=7)
        
        return df

    def calculate_rolling_features(self, df):
        """Calcula promedios móviles, máximos, mínimos y eficiencia."""
        logger.info("Calculando variables avanzadas (Rolling, Min, Max, PPM)...")
        
        if 'player_id' not in df.columns:
            logger.error(f"Columna 'player_id' no encontrada. Columnas disponibles: {list(df.columns)}")
            return df

        df = df.sort_values(by=['player_id', 'game_date'])
        
        # --- NUEVO: Limpieza de Minutos (Convertir "24:30" a float 24.5) ---
        def clean_minutes(x):
            if isinstance(x, str):
                if ':' in x:
                    try:
                        parts = x.split(':')
                        return float(parts[0]) + float(parts[1])/60
                    except:
                        return 0.0
            return float(x) if x else 0.0

        if 'min' in df.columns:
            df['min'] = df['min'].apply(clean_minutes)

        # --- NUEVO: Feature de Eficiencia (Puntos por Minuto - PPM) ---
        # Si juega 10 mins y mete 5 pts, su PPM es 0.5. Esto ayuda a proyectar.
        if 'pts' in df.columns and 'min' in df.columns:
            df['ppm'] = df['pts'] / df['min'].replace(0, np.nan)
            df['ppm'] = df['ppm'].fillna(0)

        # Agregamos PPM a las columnas objetivo
        target_cols = ['pts', 'reb', 'ast', 'min', 'ppm']
        existing_cols = [c for c in target_cols if c in df.columns]
        
        if not existing_cols:
            logger.warning(f"No se encontraron stats para procesar. Buscaba: {target_cols}")
            return df

        # Bucle de Features
        for col in existing_cols:
            # Aseguramos numérico
            df[col] = pd.to_numeric(df[col], errors='coerce') 
            grouped = df.groupby('player_id')[col]

            # 1. Promedios (Media)
            df[f'{col}_last_5'] = grouped.transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
            df[f'{col}_last_10'] = grouped.transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
            
            # 2. Riesgo (Desviación Estándar)
            df[f'{col}_std_10'] = grouped.transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).std())

            # 3. --- NUEVO: Techo (Max) ---
            # ¿Cuál fue su mejor actuación reciente? (Detecta potencial explosivo)
            df[f'{col}_max_10'] = grouped.transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).max())

            # 4. --- NUEVO: Suelo (Min) ---
            # ¿Cuál fue su peor actuación? (Detecta riesgo de piso bajo)
            df[f'{col}_min_10'] = grouped.transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).min())

        return df

    def add_opponent_stats(self, df):
        """Placeholder para futuras stats de defensa rival"""
        return df

    def run(self):
        # 1. Cargar
        df = self.load_all_stats()
        if df.empty: return None
        
        # 2. Contexto (Home/Away, Rest)
        df = self.add_context_features(df)
        
        # 3. Rolling Features (Min, Max, Mean, PPM)
        df = self.calculate_rolling_features(df)
        
        output_path = self.output_dir / "player_features_v2.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Features V2 guardadas en: {output_path}")
        logger.info(f"Total Columnas: {len(df.columns)}")
        
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    df = engineer.run()
    
    if df is not None:
        print("\n--- Ejemplo de Features para un Jugador ---")
        try:
            player_id = df['player_id'].iloc[0]
            # Mostramos las nuevas columnas Max y Min para verificar
            cols_to_show = ['game_date', 'player_name', 'pts', 'pts_last_5', 'pts_max_10', 'ppm']
            valid_cols = [c for c in cols_to_show if c in df.columns]
            print(df[df['player_id'] == player_id][valid_cols].head(10))
        except Exception as e:
            print(f"No se pudo mostrar la vista previa: {e}")