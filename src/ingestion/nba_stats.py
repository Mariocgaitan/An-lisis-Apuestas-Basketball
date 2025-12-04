import pandas as pd
import time
from datetime import datetime, timedelta
# CAMBIO AQUÍ: Usamos boxscoretraditionalv3 en lugar de v2
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv3
from src.config import RAW_DIR, LOGS_DIR

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NBADataIngestor:
    def __init__(self):
        self.save_path = RAW_DIR / "nba_stats"
        self.save_path.mkdir(parents=True, exist_ok=True)

    def get_games_on_date(self, date_str):
        try:
            board = scoreboardv2.ScoreboardV2(game_date=date_str)
            games_df = board.game_header.get_data_frame()
            return games_df['GAME_ID'].unique().tolist()
        except Exception as e:
            logger.error(f"Error obteniendo juegos para {date_str}: {e}")
            return []

    def get_boxscores(self, game_ids, date_str):
        all_stats = []
        logger.info(f"Procesando {len(game_ids)} partidos para la fecha {date_str}...")
        
        for game_id in game_ids:
            try:
                # CAMBIO AQUÍ: Usamos V3
                box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
                player_stats = box.player_stats.get_data_frame()
                
                player_stats['GAME_ID'] = game_id
                player_stats['GAME_DATE'] = date_str
                
                all_stats.append(player_stats)
                
                # Un poco de pausa es buena educación con la API
                time.sleep(0.6) 
                
            except Exception as e:
                logger.error(f"Falló descarga del GameID {game_id}: {e}")
        
        if all_stats:
            return pd.concat(all_stats, ignore_index=True)
        return pd.DataFrame()

    def run_daily_update(self, target_date=None):
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
        logger.info(f"=== Iniciando Ingesta para {target_date} ===")
        
        game_ids = self.get_games_on_date(target_date)
        
        if not game_ids:
            logger.warning("No se encontraron partidos para esta fecha.")
            return
            
        daily_stats_df = self.get_boxscores(game_ids, target_date)
        
        if not daily_stats_df.empty:
            file_name = f"{target_date}_player_stats.parquet"
            full_path = self.save_path / file_name
            
            daily_stats_df.to_parquet(full_path, index=False)
            logger.info(f"Guardado exitoso: {full_path} con {len(daily_stats_df)} registros.")
        else:
            logger.warning("Se encontraron partidos pero no stats.")

if __name__ == "__main__":
    ingestor = NBADataIngestor()
    # Volvemos a probar para verificar que desaparece la advertencia
    ingestor.run_daily_update(target_date="2024-11-01")