import json
import pandas as pd
from pathlib import Path
from glob import glob
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

class OddsProcessor:
    def __init__(self):
        self.input_dir = RAW_DIR / "odds"
        self.output_dir = PROCESSED_DIR / "odds"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_file(self):
        """Busca el archivo JSON más reciente en la carpeta raw/odds"""
        files = list(self.input_dir.glob("*.json"))
        if not files:
            return None
        # Ordenar por fecha de modificación y tomar el último
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return latest_file

    def american_to_prob(self, odd):
        """Convierte momio americano (-110, +200) a probabilidad implícita (0.52, 0.33)"""
        if odd > 0:
            return 100 / (odd + 100)
        else:
            return (-odd) / (-odd + 100)

    def process_json(self, file_path):
        """Convierte el JSON jerárquico en un DataFrame plano"""
        logger.info(f"Procesando archivo: {file_path.name}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        flat_data = []

        # Recorremos el árbol del JSON
        for game in data:
            game_id = game.get('id')
            sport = game.get('sport_key')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence_time = game.get('commence_time') # Hora del partido

            # Iteramos sobre las casas de apuestas (Pinnacle, DraftKings, Bet365...)
            for bookmaker in game.get('bookmakers', []):
                bookie_key = bookmaker.get('key')
                bookie_title = bookmaker.get('title')
                last_update = bookmaker.get('last_update')

                # Iteramos sobre los mercados (H2H, Spreads, Totals)
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key') # h2h, spreads, player_points
                    
                    # Iteramos sobre los resultados posibles
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name') # Nombre del equipo o jugador
                        price = outcome.get('price') # El Momio Americano
                        point = outcome.get('point') # Puntos (para Over/Under o Handicap)
                        
                        # Calculamos probabilidad implícita al vuelo
                        prob = self.american_to_prob(price)

                        # Agregamos la fila a nuestra lista
                        flat_data.append({
                            'game_id': game_id,
                            'commence_time': commence_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookie_key,
                            'market': market_key,
                            'selection': name, # Equipo o Jugador
                            'line': point,     # Ej: -5.5 o 220.5 (puede ser None en Moneyline)
                            'odds': price,
                            'implied_prob': round(prob, 4),
                            'last_update': last_update
                        })

        # Convertimos a DataFrame
        df = pd.DataFrame(flat_data)
        
        if not df.empty:
            # Guardamos en Parquet para eficiencia
            output_filename = file_path.stem + "_clean.parquet"
            output_path = self.output_dir / output_filename
            df.to_parquet(output_path, index=False)
            logger.info(f"Procesamiento exitoso. Guardado en: {output_path}")
            logger.info(f"Dimensiones: {df.shape} (Filas, Columnas)")
            return df
        else:
            logger.warning("El archivo JSON estaba vacío o no tenía estructura válida.")
            return None

if __name__ == "__main__":
    processor = OddsProcessor()
    
    # 1. Encontrar el archivo que acabas de descargar
    latest_file = processor.get_latest_file()
    
    if latest_file:
        # 2. Procesarlo
        df = processor.process_json(latest_file)
        
        if df is not None:
            print("\n--- Vista Previa de los Datos (Primeras 5 filas) ---")
            print(df.head())
            print("\n--- Ejemplo de Comparación (mismo partido, diferentes casas) ---")
            # Un filtro rápido para que veas la utilidad
            print(df[['home_team', 'bookmaker', 'selection', 'odds', 'implied_prob']].head(10))
    else:
        logger.error("No se encontraron archivos JSON en data/raw/odds/")