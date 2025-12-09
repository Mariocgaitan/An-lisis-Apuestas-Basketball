"""
Market Features - Entrenamiento
Cruza datos de jugadores con líneas históricas de Vegas (O/U y Spread)
para agregar contexto de mercado al modelo durante el entrenamiento.

Fuente de Odds: odds_history_clean.parquet (CSV histórico procesado)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DIR, LOGS_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketFeaturesEngineer:
    """
    Agrega features de mercado (Vegas lines) a los datos de jugadores.
    
    SOLO PARA ENTRENAMIENTO - Usa datos históricos de odds.
    
    Variables nuevas:
    - vegas_total: O/U del partido (puntos totales esperados)
    - vegas_spread: Spread del partido (diferencia esperada)
    - is_favorite: 1 si el equipo es favorito, 0 si no
    - expected_blowout: 1 si |spread| > 10 (riesgo de minutos reducidos)
    """
    
    # Mapeo de nombres: odds_history (largo) -> player_features (corto)
    TEAM_NAME_MAP = {
        'Los Angeles Lakers': 'Lakers',
        'LA Clippers': 'Clippers',
        'Golden State Warriors': 'Warriors',
        'Boston Celtics': 'Celtics',
        'Miami Heat': 'Heat',
        'Milwaukee Bucks': 'Bucks',
        'Phoenix Suns': 'Suns',
        'Dallas Mavericks': 'Mavericks',
        'Denver Nuggets': 'Nuggets',
        'Philadelphia 76ers': '76ers',
        'Brooklyn Nets': 'Nets',
        'New York Knicks': 'Knicks',
        'Chicago Bulls': 'Bulls',
        'Atlanta Hawks': 'Hawks',
        'Cleveland Cavaliers': 'Cavaliers',
        'Toronto Raptors': 'Raptors',
        'Minnesota Timberwolves': 'Timberwolves',
        'New Orleans Pelicans': 'Pelicans',
        'Sacramento Kings': 'Kings',
        'San Antonio Spurs': 'Spurs',
        'Oklahoma City Thunder': 'Thunder',
        'Portland Trail Blazers': 'Trail Blazers',
        'Utah Jazz': 'Jazz',
        'Memphis Grizzlies': 'Grizzlies',
        'Indiana Pacers': 'Pacers',
        'Detroit Pistons': 'Pistons',
        'Charlotte Hornets': 'Hornets',
        'Orlando Magic': 'Magic',
        'Washington Wizards': 'Wizards',
        'Houston Rockets': 'Rockets',
    }
    
    def __init__(self):
        # Input: datos de jugadores con pace
        self.features_path = PROCESSED_DIR / "features" / "player_features_v2_pace.parquet"
        # Input: odds históricas procesadas
        self.odds_path = PROCESSED_DIR / "odds_history_clean.parquet"
        # Output
        self.output_path = PROCESSED_DIR / "features" / "player_features_v2_market.parquet"
    
    def load_player_features(self) -> pd.DataFrame:
        """Carga los features de jugadores existentes"""
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"No se encontró {self.features_path.name}. "
                "Corre features.py y team_stats.py primero."
            )
        
        df = pd.read_parquet(self.features_path)
        df['game_date'] = pd.to_datetime(df['game_date'])
        logger.info(f"Cargados {len(df)} registros de jugadores")
        logger.info(f"Rango de fechas jugadores: {df['game_date'].min()} a {df['game_date'].max()}")
        
        return df
    
    def load_historical_odds(self) -> pd.DataFrame:
        """Carga las odds históricas del parquet procesado"""
        if not self.odds_path.exists():
            raise FileNotFoundError(
                f"No se encontró {self.odds_path.name}. "
                "Corre historical_odds.py primero para procesar el CSV."
            )
        
        odds_df = pd.read_parquet(self.odds_path)
        odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])
        
        logger.info(f"Cargadas {len(odds_df)} líneas de odds históricas")
        logger.info(f"Rango de fechas odds: {odds_df['game_date'].min()} a {odds_df['game_date'].max()}")
        
        return odds_df
    
    def map_team_names(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mapea nombres de equipos del CSV histórico al formato corto de NBA Stats.
        odds_history tiene: 'Los Angeles Lakers'
        player_features tiene: 'Lakers'
        """
        odds_df = odds_df.copy()
        odds_df['team_name_short'] = odds_df['team_name'].map(self.TEAM_NAME_MAP)
        
        # Si no está en el mapa, intentar extraer el nombre corto automáticamente
        # Ej: "Los Angeles Lakers" -> "Lakers" (última palabra)
        mask = odds_df['team_name_short'].isna()
        if mask.any():
            odds_df.loc[mask, 'team_name_short'] = odds_df.loc[mask, 'team_name'].apply(
                lambda x: x.split()[-1] if pd.notna(x) else x
            )
        
        # Verificar equipos no mapeados
        still_missing = odds_df[odds_df['team_name_short'].isna()]['team_name'].unique()
        if len(still_missing) > 0:
            logger.warning(f"Equipos sin mapear en odds: {still_missing}")
        
        return odds_df
    
    def merge_with_players(self, df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cruza los datos de jugadores con las líneas de Vegas históricas.
        El cruce es por fecha + equipo.
        """
        # Mapear nombres de equipos en odds
        odds_df = self.map_team_names(odds_df)
        
        # Identificar columna de equipo en player features
        team_col = None
        for col in ['teamname', 'team_name']:
            if col in df.columns:
                team_col = col
                break
        
        if team_col is None:
            raise ValueError("No se encontró columna de nombre de equipo en player_features")
        
        logger.info(f"Columna de equipo detectada: '{team_col}'")
        
        # Ver ejemplos de nombres para debug
        player_teams = df[team_col].dropna().unique()[:5]
        odds_teams = odds_df['team_name_short'].dropna().unique()[:5]
        logger.info(f"Ejemplos equipos jugadores: {list(player_teams)}")
        logger.info(f"Ejemplos equipos odds: {list(odds_teams)}")
        
        # Preparar odds para merge
        odds_for_merge = odds_df[['game_date', 'team_name_short', 'vegas_spread', 'vegas_total']].copy()
        odds_for_merge = odds_for_merge.drop_duplicates(subset=['game_date', 'team_name_short'])
        
        # Merge
        df_merged = pd.merge(
            df,
            odds_for_merge,
            left_on=['game_date', team_col],
            right_on=['game_date', 'team_name_short'],
            how='left'
        )
        
        # Limpiar columna auxiliar
        if 'team_name_short' in df_merged.columns:
            df_merged.drop(columns=['team_name_short'], inplace=True)
        
        # Contar cuántos registros hicieron match
        matched = df_merged['vegas_spread'].notna().sum()
        total = len(df_merged)
        match_pct = (matched / total) * 100
        logger.info(f"Match de odds: {matched}/{total} ({match_pct:.1f}%)")
        
        # Llenar valores faltantes con promedios históricos
        df_merged['vegas_total'] = df_merged['vegas_total'].fillna(210.0)  # Promedio histórico NBA
        df_merged['vegas_spread'] = df_merged['vegas_spread'].fillna(0.0)
        
        # === Features derivadas ===
        # is_favorite: 1 si spread negativo (favorito), 0 si positivo (underdog)
        df_merged['is_favorite'] = (df_merged['vegas_spread'] < 0).astype(int)
        
        # expected_blowout: 1 si |spread| > 10 (riesgo de garbage time)
        df_merged['expected_blowout'] = (df_merged['vegas_spread'].abs() > 10).astype(int)
        df_merged['implied_team_score'] = (df_merged['vegas_total'] / 2) - (df_merged['vegas_spread'] / 2)
        
        logger.info(f"✅ Features de mercado agregadas: vegas_total, vegas_spread, is_favorite, expected_blowout")
        
        return df_merged
    
    def run(self) -> pd.DataFrame:
        """Pipeline principal para entrenamiento"""
        logger.info("="*60)
        logger.info("Market Features Engineering - MODO ENTRENAMIENTO")
        logger.info("="*60)
        
        # 1. Cargar datos de jugadores
        df = self.load_player_features()
        
        # 2. Cargar odds históricas
        odds_df = self.load_historical_odds()
        
        # 3. Cruzar con jugadores
        df_final = self.merge_with_players(df, odds_df)
        
        # 4. Guardar
        df_final.to_parquet(self.output_path, index=False)
        logger.info(f"Guardado: {self.output_path}")
        logger.info(f"Total filas: {len(df_final)} | Total columnas: {len(df_final.columns)}")
        
        # Resumen de features de mercado
        market_features = ['vegas_total', 'vegas_spread', 'is_favorite', 'expected_blowout', 'implied_team_score']
        logger.info(f"🎰 Features de mercado disponibles: {market_features}")
        
        return df_final


if __name__ == "__main__":
    engineer = MarketFeaturesEngineer()
    df = engineer.run()
    
    if df is not None and not df.empty:
        print("\n" + "="*60)
        print("RESUMEN DE MARKET FEATURES")
        print("="*60)
        
        # Vista previa
        cols_to_show = ['game_date', 'player_name', 'teamname', 'vegas_total', 'vegas_spread', 'is_favorite', 'expected_blowout', 'implied_team_score']
        valid_cols = [c for c in cols_to_show if c in df.columns]
        
        if valid_cols:
            print("\n--- Vista Previa (5 registros) ---")
            print(df[valid_cols].head())
        
        # Estadísticas
        print("\n--- Estadísticas de Vegas Lines ---")
        if 'vegas_total' in df.columns:
            real_totals = df[df['vegas_total'] != 210.0]['vegas_total']
            print(f"O/U promedio (reales): {real_totals.mean():.1f}")
            print(f"O/U rango: {real_totals.min():.1f} - {real_totals.max():.1f}")
        
        if 'vegas_spread' in df.columns:
            real_spreads = df[df['vegas_spread'] != 0.0]['vegas_spread']
            print(f"Spread promedio (reales): {real_spreads.mean():.1f}")
            print(f"% Favoritos: {df['is_favorite'].mean()*100:.1f}%")
            print(f"% Expected Blowouts (|spread|>10): {df['expected_blowout'].mean()*100:.1f}%")
        
        # Cobertura
        print("\n--- Cobertura de Datos ---")
        total = len(df)
        with_odds = len(df[df['vegas_spread'] != 0.0])
        print(f"Registros con odds reales: {with_odds}/{total} ({with_odds/total*100:.1f}%)")