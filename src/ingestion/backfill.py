import time
from datetime import datetime, timedelta
from tqdm import tqdm
from src.ingestion.nba_stats import NBADataIngestor
from src.config import LOGS_DIR

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "backfill.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_backfill_range(start_date: str, end_date: str):
    """
    Descarga datos de jugadores entre dos fechas específicas.
    
    Args:
        start_date: Fecha inicio en formato 'YYYY-MM-DD'
        end_date: Fecha fin en formato 'YYYY-MM-DD'
    """
    ingestor = NBADataIngestor()
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    logger.info(f"=== INICIANDO BACKFILL ===")
    logger.info(f"Desde: {start_date} Hasta: {end_date}")
    
    # Generar lista de fechas
    dates_to_process = []
    current = start
    while current <= end:
        dates_to_process.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    logger.info(f"Total días a procesar: {len(dates_to_process)}")
    
    # Procesar con barra de progreso
    successful = 0
    failed = 0
    
    for date_str in tqdm(dates_to_process, desc="Descargando"):
        try:
            ingestor.run_daily_update(target_date=date_str)
            successful += 1
            # Pausa para no saturar la API
            time.sleep(1.5)
            
        except Exception as e:
            logger.error(f"Error en {date_str}: {e}")
            failed += 1
    
    logger.info(f"=== BACKFILL COMPLETADO ===")
    logger.info(f"Exitosos: {successful} | Fallidos: {failed}")


def run_backfill(days_back=30):
    """
    Descarga datos desde hace 'days_back' días hasta ayer.
    (Función original para compatibilidad)
    """
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)
    
    run_backfill_range(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


if __name__ == "__main__":
    # === CONFIGURACIÓN PARA ENTRENAR CON ODDS HISTÓRICAS ===
    # Tu CSV de odds va de 2007 a 2023-01-16
    # Descargamos la temporada 2022-23 (Oct 2022 - Ene 2023)
    
    print("="*60)
    print("BACKFILL - Temporada 2022-23")
    print("Para coincidir con odds históricas disponibles")
    print("="*60)
    
    # Descarga ~3 meses de datos (Oct-Dic 2022 + Ene 2023)
    # Esto tomará tiempo (~1-2 horas dependiendo de tu conexión)
    run_backfill_range(
        start_date="2022-10-28",  # Inicio temporada NBA 2022-23
        end_date="2022-12-25"     # Última fecha en tu CSV de odds
    )