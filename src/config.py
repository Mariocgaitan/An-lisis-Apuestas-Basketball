import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# --- Rutas Base ---
# Esto hace que el código funcione igual en tu laptop o en un servidor
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"

# Asegurar que los directorios existan
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuración de APIs ---
# Si no encuentra la key, lanza error para avisarte rápido
ODDS_API_KEY = os.getenv("ODDS_API_KEY") 

# --- Configuración de Descarga ---
# Temporadas a descargar para el historial (formatos de nba_api)
SEASONS_TO_FETCH = ["2022-23", "2023-24", "2024-25"]