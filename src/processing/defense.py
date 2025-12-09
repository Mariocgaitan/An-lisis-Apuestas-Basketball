import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DIR, LOGS_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefenseEngineer:
    def __init__(self):
        self.features_path = PROCESSED_DIR / "features" / "player_features_v2.parquet"
        self.output_path = PROCESSED_DIR / "features" / "player_features_v3_defense.parquet"
        

    def run(self):
        logger.info("Cargando datos para cálculo defensivo...")
        if not self.features_path.exists():
            raise FileNotFoundError("No se encontró player_features_v2.parquet")
        
        df = pd.read_parquet(self.features_path)
        
        # --- PASO 1: Calcular Puntos Totales por Equipo en cada Juego ---
        # Agrupamos por Juego y Equipo para saber cuánto anotó cada uno
        # Usamos 'pts' (suma de puntos de todos los jugadores de ese equipo en ese juego)
        team_stats = df.groupby(['game_id', 'team_id', 'game_date'])['pts'].sum().reset_index()
        team_stats.rename(columns={'pts': 'team_score'}, inplace=True)
        
        # --- PASO 2: Identificar al Rival y sus Puntos Permitidos ---
        # Un juego tiene 2 equipos. Si yo soy el equipo A, el equipo B es mi rival.
        # Hacemos un self-merge por game_id
        games = pd.merge(team_stats, team_stats, on='game_id', suffixes=('', '_opp'))
        
        # Filtramos para que no se cruce el equipo consigo mismo
        games = games[games['team_id'] != games['team_id_opp']]
        
        # Ahora tenemos:
        # team_id: Mi equipo
        # team_id_opp: El rival
        # team_score_opp: Los puntos que metió el rival (o sea, los que YO permití)
        
        # Seleccionamos columnas clave para calcular la defensa del rival
        # Queremos saber: "Cuando juego contra team_id_opp, ¿cuántos puntos suele permitir?"
        defense_df = games[['game_date', 'team_id_opp', 'team_score']].copy()
        defense_df.rename(columns={
            'team_id_opp': 'team_id', # Para agrupar por el equipo que defiende
            'team_score': 'points_allowed' # Los puntos que le metieron
        }, inplace=True)
        
        # --- PASO 3: Rolling Defense Stats (Promedio de puntos permitidos últimos 10 juegos) ---
        defense_df = defense_df.sort_values(['team_id', 'game_date'])
        
        defense_df['opp_pts_allowed_last_10'] = defense_df.groupby('team_id')['points_allowed'].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
        )
        
        # Limpieza: Nos quedamos solo con Fecha, Equipo y su Ranking Defensivo
        defense_metrics = defense_df[['game_date', 'team_id', 'opp_pts_allowed_last_10']].drop_duplicates()
        
        logger.info("Métricas defensivas calculadas. Cruzando con stats de jugadores...")
        
        # --- PASO 4: Cruzar de vuelta con la tabla de Jugadores ---
        # Necesitamos saber contra quién jugó cada jugador.
        # Volvemos a usar la lógica de game_id para pegar el opp_id al df original
        
        # Primero, obtenemos el Opponent ID para cada fila del dataframe original
        game_opponents = games[['game_id', 'team_id', 'team_id_opp']]
        
        df_with_opp = pd.merge(df, game_opponents, on=['game_id', 'team_id'], how='left')
        
        # Ahora pegamos las stats defensivas usando el 'team_id_opp'
        final_df = pd.merge(
            df_with_opp,
            defense_metrics,
            left_on=['team_id_opp', 'game_date'],
            right_on=['team_id', 'game_date'],
            suffixes=('', '_defense_merge'),
            how='left'
        )
        
        # Limpieza final de columnas duplicadas
        final_df.drop(columns=['team_id_defense_merge'], inplace=True)
        
        # Llenar nulos (al principio de temporada no hay historia) con el promedio de la liga (aprox 114 pts)
        final_df['opp_pts_allowed_last_10'] = final_df['opp_pts_allowed_last_10'].fillna(114.0)
        
        # Guardar V3
        final_df.to_parquet(self.output_path, index=False)
        logger.info(f"Guardado exitoso: {self.output_path}")
        logger.info("Nueva variable disponible: 'opp_pts_allowed_last_10'")
        
        return final_df

if __name__ == "__main__":
    eng = DefenseEngineer()
    eng.run()