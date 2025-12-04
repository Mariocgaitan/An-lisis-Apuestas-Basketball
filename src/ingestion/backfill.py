import time
from datetime import datetime, timedelta
from tqdm import tqdm # Barra de progreso para que no te desesperes
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

def run_backfill(days_back=30):
    """
    Descarga datos desde hace 'days_back' días hasta ayer.
    """
    ingestor = NBADataIngestor()
    
    # Fecha final: Ayer
    end_date = datetime.now() - timedelta(days=1)
    # Fecha inicial: Hoy menos X días
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"=== INICIANDO BACKFILL DE {days_back} DÍAS ===")
    logger.info(f"Desde: {start_date.strftime('%Y-%m-%d')} Hasta: {end_date.strftime('%Y-%m-%d')}")
    
    # Generamos la lista de fechas
    current_date = start_date
    dates_to_process = []
    while current_date <= end_date:
        dates_to_process.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    # Usamos tqdm para mostrar una barra de progreso en la terminal
    for date_str in tqdm(dates_to_process, desc="Descargando Historia"):
        try:
            # Reutilizamos tu lógica diaria
            ingestor.run_daily_update(target_date=date_str)
            
            # PAUSA DE SEGURIDAD: 
            # Como vamos a hacer muchas peticiones seguidas, descansamos 1.5 segundos entre días
            # para evitar el Error 429 (Too Many Requests).
            time.sleep(1.5) 
            
        except Exception as e:
            logger.error(f"Error crítico en fecha {date_str}: {e}")

    logger.info("=== BACKFILL COMPLETADO ===")

if __name__ == "__main__":
    # Puedes cambiar el número de días. 
    # 30 días es bueno para empezar rápido.
    # 90 días sería toda la temporada actual aprox.
    run_backfill(days_back=30)