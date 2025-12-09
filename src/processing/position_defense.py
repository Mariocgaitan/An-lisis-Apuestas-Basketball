"""
Fase 2: Defensa por Posición

Features:
- opp_pts_allowed_to_pos_last_10: Puntos que el rival permite a cada posición (últimos 10 juegos)
- def_rating_last_5: Rating defensivo del rival (últimos 5 juegos)

Requiere: Columna 'position' en los datos de jugadores
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DIR, RAW_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionDefenseFeatures:
    """
    Calcula features de defensa basadas en la posición del jugador.
    """
    
    # Mapeo de posiciones específicas a las 5 posiciones base
    # La API V3 usa: G, F, C, G-F, F-G, F-C, C-F
    POSITION_MAP = {
        'G': 'G',      # Guard
        'F': 'F',      # Forward
        'C': 'C',      # Center
        'G-F': 'G',    # Guard-Forward → Guard
        'F-G': 'F',    # Forward-Guard → Forward
        'F-C': 'F',    # Forward-Center → Forward
        'C-F': 'C',    # Center-Forward → Center
        '': 'G',       # Default
    }
    
    def __init__(self):
        self.features_path = PROCESSED_DIR / "features" / "player_features_v2_market.parquet"
    
    def add_position_defense_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega features de defensa por posición.
        """
        if 'position' not in df.columns:
            logger.warning("Columna 'position' no encontrada. Saltando features de posición.")
            return df
        
        logger.info("Calculando features de defensa por posición...")
        
        # Normalizar posiciones
        df['position'] = df['position'].fillna('').astype(str)
        df['position_base'] = df['position'].map(self.POSITION_MAP).fillna('G')
        
        # Ordenar por fecha
        df = df.sort_values(['game_date', 'game_id', 'player_id']).reset_index(drop=True)
        
        # 1. Calcular puntos permitidos por posición por equipo (rolling)
        df = self._calc_pts_allowed_to_position(df)
        
        # 2. Calcular defensive rating del rival (últimos 5)
        df = self._calc_def_rating_last_5(df)
        
        return df
    
    def _calc_pts_allowed_to_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada partido, calcula cuántos puntos permite cada equipo
        a cada posición en los últimos 10 juegos.
        
        Lógica:
        1. Para cada juego, sumar pts por posición para cada equipo
        2. Invertir: los pts que anotó el equipo A a la posición G son los pts que el equipo B permitió a G
        3. Rolling 10 juegos por (equipo_defensor, posición)
        """
        # Paso 1: Calcular pts por posición por equipo en cada juego
        game_pos_pts = df.groupby(['game_date', 'game_id', 'team_id', 'position_base']).agg({
            'pts': 'sum'
        }).reset_index()
        game_pos_pts = game_pos_pts.rename(columns={'pts': 'pts_scored_by_pos'})
        
        # Paso 2: Obtener el rival de cada equipo en cada juego
        # Crear mapeo game_id -> [team_1, team_2]
        teams_per_game = df.groupby('game_id')['team_id'].apply(lambda x: list(x.unique())).to_dict()
        
        def get_opponent(row):
            teams = teams_per_game.get(row['game_id'], [])
            if len(teams) == 2:
                return teams[1] if row['team_id'] == teams[0] else teams[0]
            return None
        
        game_pos_pts['team_id_defender'] = game_pos_pts.apply(get_opponent, axis=1)
        
        # Eliminar filas sin rival (juegos con datos incompletos)
        game_pos_pts = game_pos_pts.dropna(subset=['team_id_defender'])
        
        # Paso 3: Ahora pts_scored_by_pos del team_id son los pts PERMITIDOS por team_id_defender
        # Renombrar para claridad
        pts_allowed = game_pos_pts[['game_date', 'game_id', 'team_id_defender', 'position_base', 'pts_scored_by_pos']].copy()
        pts_allowed = pts_allowed.rename(columns={
            'team_id_defender': 'team_id',
            'pts_scored_by_pos': 'pts_allowed_to_pos'
        })
        
        # Paso 4: Rolling 10 juegos por (equipo, posición)
        pts_allowed = pts_allowed.sort_values(['team_id', 'position_base', 'game_date'])
        
        pts_allowed['pts_allowed_to_pos_last_10'] = pts_allowed.groupby(
            ['team_id', 'position_base']
        )['pts_allowed_to_pos'].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )
        
        # Paso 5: Merge con df original
        # Necesitamos unir: para cada jugador, buscar cuánto permite su RIVAL a su posición
        # df tiene team_id del jugador, necesitamos team_id_opp
        
        # Agregar team_id_opp si no existe
        if 'team_id_opp' not in df.columns:
            df['team_id_opp'] = df.apply(
                lambda row: get_opponent({'game_id': row['game_id'], 'team_id': row['team_id']}),
                axis=1
            )
        
        # Crear tabla de lookup: (game_date, team_id, position) -> pts_allowed_to_pos_last_10
        lookup = pts_allowed[['game_date', 'game_id', 'team_id', 'position_base', 'pts_allowed_to_pos_last_10']].copy()
        lookup = lookup.rename(columns={'team_id': 'team_id_opp'})
        
        # Merge: buscar cuánto permite el team_id_opp a la position_base del jugador
        df = df.merge(
            lookup[['game_id', 'team_id_opp', 'position_base', 'pts_allowed_to_pos_last_10']],
            on=['game_id', 'team_id_opp', 'position_base'],
            how='left'
        )
        
        # Renombrar para claridad
        df = df.rename(columns={'pts_allowed_to_pos_last_10': 'opp_pts_to_pos_last_10'})
        
        logger.info(f"✅ Feature añadida: opp_pts_to_pos_last_10")
        logger.info(f"   Cobertura: {df['opp_pts_to_pos_last_10'].notna().sum()}/{len(df)} ({df['opp_pts_to_pos_last_10'].notna().mean()*100:.1f}%)")
        
        return df
    
    def _calc_def_rating_last_5(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el Defensive Rating del equipo rival en últimos 5 juegos.
        DEF_RTG aproximado = Puntos permitidos promedio (sin posesiones exactas)
        """
        # Puntos totales permitidos por equipo por juego
        if 'team_id_opp' not in df.columns:
            logger.warning("team_id_opp no disponible. Saltando def_rating_last_5.")
            return df
        
        # Calcular puntos totales por equipo por juego
        team_pts_per_game = df.groupby(['game_date', 'game_id', 'team_id']).agg({
            'pts': 'sum'
        }).reset_index()
        team_pts_per_game = team_pts_per_game.rename(columns={'pts': 'team_total_pts'})
        
        # Los puntos del team_id son los puntos PERMITIDOS por el rival
        teams_per_game = df.groupby('game_id')['team_id'].apply(lambda x: list(x.unique())).to_dict()
        
        def get_opponent_simple(row):
            teams = teams_per_game.get(row['game_id'], [])
            if len(teams) == 2:
                return teams[1] if row['team_id'] == teams[0] else teams[0]
            return None
        
        team_pts_per_game['team_id_defender'] = team_pts_per_game.apply(get_opponent_simple, axis=1)
        team_pts_per_game = team_pts_per_game.dropna(subset=['team_id_defender'])
        
        # Renombrar: team_total_pts son los pts PERMITIDOS por team_id_defender
        def_stats = team_pts_per_game[['game_date', 'game_id', 'team_id_defender', 'team_total_pts']].copy()
        def_stats = def_stats.rename(columns={
            'team_id_defender': 'team_id',
            'team_total_pts': 'pts_allowed'
        })
        
        # Rolling 5 juegos
        def_stats = def_stats.sort_values(['team_id', 'game_date'])
        def_stats['def_rating_last_5'] = def_stats.groupby('team_id')['pts_allowed'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
        )
        
        # Crear lookup para merge
        lookup = def_stats[['game_id', 'team_id', 'def_rating_last_5']].copy()
        lookup = lookup.rename(columns={'team_id': 'team_id_opp'})
        
        # Merge
        df = df.merge(
            lookup[['game_id', 'team_id_opp', 'def_rating_last_5']],
            on=['game_id', 'team_id_opp'],
            how='left'
        )
        
        # Renombrar para claridad (es el def rating del OPONENTE)
        df = df.rename(columns={'def_rating_last_5': 'opp_def_rating_last_5'})
        
        logger.info(f"✅ Feature añadida: opp_def_rating_last_5")
        logger.info(f"   Cobertura: {df['opp_def_rating_last_5'].notna().sum()}/{len(df)} ({df['opp_def_rating_last_5'].notna().mean()*100:.1f}%)")
        
        return df


def add_position_features_to_dataset():
    """
    Pipeline para añadir features de posición al dataset existente.
    """
    processor = PositionDefenseFeatures()
    
    # Cargar dataset actual
    logger.info(f"Cargando: {processor.features_path}")
    df = pd.read_parquet(processor.features_path)
    logger.info(f"Registros cargados: {len(df)}")
    
    # Verificar si tiene posición
    if 'position' not in df.columns:
        logger.error("❌ El dataset no tiene columna 'position'.")
        logger.info("Necesitas regenerar features.py para que incluya 'position'.")
        logger.info("La columna existe en raw data pero no se está pasando al parquet final.")
        return None
    
    # Añadir features
    df = processor.add_position_defense_features(df)
    
    # Guardar
    output_path = PROCESSED_DIR / "features" / "player_features_v2_position.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Guardado en: {output_path}")
    logger.info(f"   Total columnas: {len(df.columns)}")
    
    # Mostrar nuevas features
    new_cols = ['position_base', 'opp_pts_to_pos_last_10', 'opp_def_rating_last_5']
    available = [c for c in new_cols if c in df.columns]
    logger.info(f"   Nuevas features Fase 2: {available}")
    
    return df


if __name__ == "__main__":
    add_position_features_to_dataset()