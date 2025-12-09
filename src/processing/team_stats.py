import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DIR, LOGS_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamStatsEngineer:
    def __init__(self):
        # Usamos el V2 que tiene las columnas estandarizadas
        self.features_path = PROCESSED_DIR / "features" / "player_features_v2.parquet"
        self.output_path = PROCESSED_DIR / "features" / "player_features_v2_pace.parquet"

    def calculate_possessions(self, row):
        """
        Fórmula de Posesiones NBA (Aproximada):
        Poss = FGA + 0.44 * FTA - ORB + TOV
        """
        # Necesitamos manejar posibles nulos con 0
        fga = row.get('fieldgoalsattempted', 0)
        fta = row.get('freethrowsattempted', 0)
        orb = row.get('reboundsoffensive', 0)
        tov = row.get('turnovers', 0)
        
        return fga + (0.44 * fta) - orb + tov

    def run(self):
        logger.info("Calculando Ritmo (Pace) y Eficiencia Defensiva...")
        
        if not self.features_path.exists():
            raise FileNotFoundError("No se encontró player_features_v2.parquet")
        
        df = pd.read_parquet(self.features_path)
        
        # --- PASO 1: Agrupar stats por Equipo y Partido ---
        # Sumamos las stats individuales para obtener las del equipo
        cols_to_sum = [
            'pts', 'fieldgoalsattempted', 'freethrowsattempted', 
            'reboundsoffensive', 'turnovers', 'min'
        ]
        
        # Aseguramos que existan las columnas (si el standardize funcionó, deberían estar)
        valid_cols = [c for c in cols_to_sum if c in df.columns]
        
        team_game_stats = df.groupby(['game_id', 'team_id', 'game_date'])[valid_cols].sum().reset_index()
        
        # --- PASO 2: Calcular Métricas Avanzadas por Equipo ---
        # Calculamos posesiones totales del equipo en ese juego
        team_game_stats['possessions'] = team_game_stats.apply(self.calculate_possessions, axis=1)
        
        # Pace = (Posesiones / Minutos) * 48
        # Nota: 'min' aquí es la suma de todos los jugadores (aprox 240 min). 
        # Dividimos entre 5 para tener minutos de partido.
        team_game_stats['team_pace'] = (team_game_stats['possessions'] / (team_game_stats['min'] / 5)) * 48
        
        # Defensive Rating (no podemos calcularlo directo sin los puntos del rival, 
        # así que primero hacemos el merge de rivales)
        
        # --- PASO 3: Identificar al Rival ---
        games = pd.merge(team_game_stats, team_game_stats, on='game_id', suffixes=('', '_opp'))
        games = games[games['team_id'] != games['team_id_opp']]
        
        # Ahora tenemos Pace del rival y Puntos que anotó el rival
        # Rival Pace: Si el rival juega rápido, me obliga a jugar rápido.
        # Rival Points: Cuántos puntos metió el rival (o sea, cuántos permití yo).
        
        stats_df = games[['game_date', 'team_id', 'team_pace', 'team_id_opp', 'team_pace_opp', 'pts_opp']].copy()
        
        # --- PASO 4: Rolling Stats (Tendencias de los últimos 10 juegos) ---
        stats_df = stats_df.sort_values(['team_id', 'game_date'])
        
        # 4.1. Ritmo Propio (¿Mi equipo viene jugando rápido?)
        stats_df['pace_last_10'] = stats_df.groupby('team_id')['team_pace'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        
        # 4.2. Ritmo del Rival (¿El equipo contra el que juego permite muchas posesiones?)
        # Para esto, necesitamos agrupar por el team_id_opp
        # Creamos un dataframe temporal de "Fuerza del Rival"
        rival_strength = stats_df[['game_date', 'team_id', 'team_pace', 'pts_opp']].copy()
        rival_strength.rename(columns={
            'team_id': 'team_id_opp', # Ahora agrupamos por el equipo rival
            'team_pace': 'opp_pace_actual',
            'pts_opp': 'points_allowed_actual' # Puntos que el rival recibió (de mi)
        }, inplace=True)
        
        rival_strength = rival_strength.sort_values(['team_id_opp', 'game_date'])
        
        # ¿A qué ritmo juega este rival usualmente?
        rival_strength['opp_pace_last_10'] = rival_strength.groupby('team_id_opp')['opp_pace_actual'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        
        # ¿Cuántos puntos permite este rival usualmente? (Defensa General)
        rival_strength['opp_pts_allowed_last_10'] = rival_strength.groupby('team_id_opp')['points_allowed_actual'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        
        # Limpiamos duplicados
        rival_stats = rival_strength[['game_date', 'team_id_opp', 'opp_pace_last_10', 'opp_pts_allowed_last_10']].drop_duplicates()
        
        # --- PASO 5: Merge Final con Jugadores ---
        logger.info("Cruzando métricas de equipo con jugadores...")
        
        # A. Pegamos stats de MI equipo (mi ritmo)
        df = pd.merge(df, stats_df[['game_date', 'team_id', 'pace_last_10']], on=['game_date', 'team_id'], how='left')
        
        # B. Pegamos stats del RIVAL (su ritmo y defensa)
        # Primero necesitamos saber quién es el rival en el df original
        # Usamos el 'matchup' o volvemos a obtenerlo de games
        game_opponents = games[['game_id', 'team_id', 'team_id_opp']].drop_duplicates()
        df = pd.merge(df, game_opponents, on=['game_id', 'team_id'], how='left')
        
        df = pd.merge(
            df, 
            rival_stats, 
            on=['game_date', 'team_id_opp'], 
            how='left'
        )
        
        # Llenar NAs iniciales
        df['pace_last_10'] = df['pace_last_10'].fillna(98.0) # Promedio liga
        df['opp_pace_last_10'] = df['opp_pace_last_10'].fillna(98.0)
        df['opp_pts_allowed_last_10'] = df['opp_pts_allowed_last_10'].fillna(114.0)
        
        df.to_parquet(self.output_path, index=False)
        logger.info(f"Guardado: {self.output_path}")
        logger.info("Nuevas Variables: pace_last_10, opp_pace_last_10, opp_pts_allowed_last_10")
        
        return df

if __name__ == "__main__":
    eng = TeamStatsEngineer()
    eng.run()