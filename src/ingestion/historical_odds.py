import pandas as pd
import numpy as np
from src.config import DATA_DIR, PROCESSED_DIR

# DICCIONARIO MAESTRO
# Mapeamos los nombres de TU csv (izquierda) a los nombres oficiales de la NBA API (derecha)
# Incluye reubicaciones históricas (Seattle -> OKC, NJ -> Brooklyn)
TEAM_MAP = {
    'Atlanta': 'Atlanta Hawks',
    'Boston': 'Boston Celtics',
    'Brooklyn': 'Brooklyn Nets', 'New Jersey': 'Brooklyn Nets', # Historia
    'Charlotte': 'Charlotte Hornets',
    'Chicago': 'Chicago Bulls',
    'Cleveland': 'Cleveland Cavaliers',
    'Dallas': 'Dallas Mavericks',
    'Denver': 'Denver Nuggets',
    'Detroit': 'Detroit Pistons',
    'Golden State': 'Golden State Warriors',
    'Houston': 'Houston Rockets',
    'Indiana': 'Indiana Pacers',
    'LA Clippers': 'LA Clippers', # nba_api a veces usa este o Los Angeles Clippers
    'LA Lakers': 'Los Angeles Lakers', # CRÍTICO: nba_api usa el nombre largo
    'Memphis': 'Memphis Grizzlies',
    'Miami': 'Miami Heat',
    'Milwaukee': 'Milwaukee Bucks',
    'Minnesota': 'Minnesota Timberwolves',
    'New Orleans': 'New Orleans Pelicans',
    'New York': 'New York Knicks',
    'Oklahoma City': 'Oklahoma City Thunder', 'Seattle': 'Oklahoma City Thunder', # Historia
    'Orlando': 'Orlando Magic',
    'Philadelphia': 'Philadelphia 76ers',
    'Phoenix': 'Phoenix Suns',
    'Portland': 'Portland Trail Blazers',
    'Sacramento': 'Sacramento Kings',
    'San Antonio': 'San Antonio Spurs',
    'Toronto': 'Toronto Raptors',
    'Utah': 'Utah Jazz',
    'Washington': 'Washington Wizards'
}

def load_and_clean_odds():
    # Ajusta el nombre si tu archivo se llama diferente
    csv_path = DATA_DIR / "external" / "NBAoddsData.csv"
    
    if not csv_path.exists():
        print(f"❌ Error: No encontré el archivo en {csv_path}")
        return

    print("Cargando CSV de Odds...")
    # Tu CSV tiene la columna 'date', pandas intentará leerla
    df = pd.read_csv(csv_path)

    # 1. Estandarización de Fecha
    # Convertimos la columna 'date' a datetime
    df['game_date'] = pd.to_datetime(df['date']).dt.normalize()

    # 2. Estandarización de Equipos
    # Usamos el mapa para traducir 'LA Lakers' -> 'Los Angeles Lakers'
    df['team_name'] = df['team'].map(TEAM_MAP)

    # Verificación de seguridad: ¿Quedó algún equipo sin traducir?
    missing_teams = df[df['team_name'].isna()]['team'].unique()
    if len(missing_teams) > 0:
        print(f"⚠️ ALERTA: Hay equipos que no se mapearon: {missing_teams}")
        # Los rellenamos con el nombre original por si acaso
        df['team_name'] = df['team_name'].fillna(df['team'])

    # 3. Selección y Renombrado
    # spread negativo (-) es favorito en tu csv, igual que en Las Vegas. Perfecto.
    clean_df = df[['game_date', 'team_name', 'spread', 'total']].copy()
    
    clean_df.rename(columns={
        'spread': 'vegas_spread',
        'total': 'vegas_total'
    }, inplace=True)

    # 4. Eliminar duplicados y nulos
    clean_df = clean_df.drop_duplicates(subset=['game_date', 'team_name'])
    clean_df = clean_df.dropna()

    # Guardar
    output_path = PROCESSED_DIR / "odds_history_clean.parquet"
    clean_df.to_parquet(output_path)
    
    print(f"✅ Odds procesados exitosamente: {output_path}")
    print(f"Total filas: {len(clean_df)}")
    print("Muestra:")
    print(clean_df.head())

if __name__ == "__main__":
    load_and_clean_odds()