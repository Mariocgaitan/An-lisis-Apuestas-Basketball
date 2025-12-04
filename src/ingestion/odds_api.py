import requests
import json
import time
from datetime import datetime
from src.config import ODDS_API_KEY, RAW_DIR, LOGS_DIR

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

class OddsApiClient:
    def __init__(self):
        self.api_key = ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.sport_key = "basketball_nba"
        
        # Carpeta para guardar los JSON de momios
        self.save_path = RAW_DIR / "odds"
        self.save_path.mkdir(parents=True, exist_ok=True)

    def check_quota(self):
        """
        Consulta el estado de tu cuenta sin gastar mucho (usa el endpoint de deportes).
        Retorna: (solicitudes_usadas, solicitudes_restantes)
        """
        try:
            url = f"{self.base_url}/?apiKey={self.api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                # Los headers de la respuesta traen la info de la cuota
                requests_remaining = response.headers.get("x-requests-remaining")
                requests_used = response.headers.get("x-requests-used")
                logger.info(f"Estado de API: Usadas: {requests_used} | Restantes: {requests_remaining}")
                return int(requests_remaining)
            else:
                logger.error(f"Error revisando cuota: {response.status_code} - {response.text}")
                return 0
        except Exception as e:
            logger.error(f"Error de conexión: {e}")
            return 0

# Agregamos 'eu' (Europa) para traer a Pinnacle y Bet365
    def get_odds(self, markets="h2h,spreads", regions="us,eu"):
        """
        Descarga los momios para la NBA.
        markets: 'h2h' (ganador), 'spreads' (hándicap), 'totals' (altas/bajas), 'player_points', etc.
        """
        # Verificamos cuota antes de disparar
        remaining = self.check_quota()
        if remaining < 10:
            logger.warning("¡ALERTA! Te quedan menos de 10 llamadas a la API. Abortando para seguridad.")
            return None

        url = f"{self.base_url}/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,        # 'us' para momios americanos (DraftKings, FanDuel, etc.)
            "markets": markets,        # Qué tipo de apuesta queremos
            "oddsFormat": "american",  # Formato +100, -110
            "dateFormat": "iso"
        }

        logger.info(f"Solicitando momios para mercados: {markets}...")
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Guardamos el JSON crudo con fecha y hora
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # Limpiamos el nombre del mercado para el archivo (ej. player_points -> player_points)
                market_name = markets.replace(",", "_")
                filename = f"odds_{market_name}_{timestamp}.json"
                
                full_path = self.save_path / filename
                with open(full_path, "w") as f:
                    json.dump(data, f, indent=4)
                
                logger.info(f"Momios guardados exitosamente en: {filename}")
                return data
            else:
                logger.error(f"Error al bajar momios: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Excepción al pedir momios: {e}")
            return None

if __name__ == "__main__":
    # PRUEBA SEGURA
    client = OddsApiClient()
    
    # 1. Primero solo revisamos si la llave funciona
    print("--- Verificando conexión ---")
    client.check_quota()
    
    # 2. Si quieres probar descargar momios reales (Gasta 1 crédito), descomenta la siguiente línea:
    client.get_odds(markets="h2h")