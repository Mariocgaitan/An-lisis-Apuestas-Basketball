import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from src.config import PROCESSED_DIR, BASE_DIR, LOGS_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EVCalculator:
    """
    Calcula Expected Value comparando predicciones del modelo vs líneas de casas de apuestas.
    
    Fórmula EV:
    EV = (Prob_Ganar * Ganancia) - (Prob_Perder * Apuesta)
    
    Si EV > 0, hay valor esperado positivo.
    """
    
    def __init__(self):
        self.model_path = BASE_DIR / "src" / "modeling" / "model_store" / "xgb_points_v1.pkl"
        self.odds_dir = PROCESSED_DIR / "odds"
        self.output_dir = BASE_DIR / "data" / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Error estándar del modelo (MAE histórico ~4-5 puntos)
        # Esto lo usamos para calcular probabilidades
        self.model_std = 5.0  # Ajustar según tu MAE real
    
    def load_latest_odds(self) -> pd.DataFrame:
        """Carga el archivo de momios más reciente"""
        files = list(self.odds_dir.glob("*_clean.parquet"))
        if not files:
            raise FileNotFoundError("No hay momios procesados. Corre odds_processor.py primero.")
        
        latest = max(files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Cargando momios: {latest.name}")
        return pd.read_parquet(latest)
    
    def american_to_decimal(self, american_odds: float) -> float:
        """Convierte momio americano a decimal para calcular ganancia"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def calculate_win_probability(self, prediction: float, line: float, over: bool = True) -> float:
        """
        Calcula la probabilidad de ganar la apuesta usando distribución normal.
        
        Args:
            prediction: Puntos predichos por el modelo
            line: Línea de la casa (ej: 25.5)
            over: True si apostamos al Over, False si al Under
        
        Returns:
            Probabilidad de ganar (0 a 1)
        """
        # Usamos distribución normal centrada en la predicción
        # con desviación estándar = error típico del modelo
        
        if over:
            # P(puntos_reales > line)
            prob = 1 - stats.norm.cdf(line, loc=prediction, scale=self.model_std)
        else:
            # P(puntos_reales < line)
            prob = stats.norm.cdf(line, loc=prediction, scale=self.model_std)
        
        return prob
    
    def calculate_ev(self, win_prob: float, decimal_odds: float, stake: float = 100) -> float:
        """
        Calcula Expected Value.
        
        EV = (P_win * Profit) - (P_lose * Stake)
        """
        profit = stake * (decimal_odds - 1)
        ev = (win_prob * profit) - ((1 - win_prob) * stake)
        return ev
    
    def calculate_kelly(self, win_prob: float, decimal_odds: float) -> float:
        """
        Calcula el % óptimo del bankroll a apostar (Kelly Criterion).
        
        Kelly = (bp - q) / b
        Donde:
            b = decimal_odds - 1 (ganancia por unidad apostada)
            p = probabilidad de ganar
            q = probabilidad de perder (1 - p)
        """
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Nunca apostar más del 25% ni valores negativos
        return max(0, min(kelly, 0.25))
    
    def analyze_player_props(self, predictions: dict) -> pd.DataFrame:
        """
        Analiza apuestas de props de jugadores.
        
        Args:
            predictions: Dict con {player_name: predicted_points}
                        Ej: {"LeBron James": 27.5, "Stephen Curry": 29.0}
        
        Returns:
            DataFrame con oportunidades de valor ordenadas por EV
        """
        odds_df = self.load_latest_odds()
        
        # Filtrar solo mercados de player_points si existen
        # (Nota: The Odds API puede no tener este mercado siempre)
        player_markets = odds_df[odds_df['market'].str.contains('player', case=False, na=False)]
        
        if player_markets.empty:
            logger.warning("No hay mercados de jugadores en los momios actuales.")
            logger.info("Analizando con datos simulados para demostración...")
            return self._demo_analysis(predictions)
        
        results = []
        
        for player, pred_pts in predictions.items():
            player_odds = player_markets[
                player_markets['selection'].str.contains(player, case=False, na=False)
            ]
            
            for _, row in player_odds.iterrows():
                line = row.get('line', 0)
                odds = row['odds']
                
                if line is None or pd.isna(line):
                    continue
                
                decimal_odds = self.american_to_decimal(odds)
                
                # Determinar si es Over o Under
                is_over = 'over' in row['selection'].lower()
                
                win_prob = self.calculate_win_probability(pred_pts, line, over=is_over)
                ev = self.calculate_ev(win_prob, decimal_odds)
                kelly = self.calculate_kelly(win_prob, decimal_odds)
                
                results.append({
                    'player': player,
                    'prediction': pred_pts,
                    'line': line,
                    'type': 'OVER' if is_over else 'UNDER',
                    'odds': odds,
                    'decimal_odds': round(decimal_odds, 3),
                    'bookmaker': row['bookmaker'],
                    'win_prob': round(win_prob, 3),
                    'implied_prob': row['implied_prob'],
                    'edge': round(win_prob - row['implied_prob'], 3),
                    'ev_per_100': round(ev, 2),
                    'kelly_pct': round(kelly * 100, 2)
                })
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            results_df = results_df.sort_values('ev_per_100', ascending=False)
        
        return results_df
    
    def _demo_analysis(self, predictions: dict) -> pd.DataFrame:
        """Análisis de demostración con líneas simuladas"""
        results = []
        
        # Simulamos líneas típicas (predicción - 2 a predicción + 2)
        for player, pred_pts in predictions.items():
            for offset in [-2.5, -0.5, 0.5, 2.5]:
                line = pred_pts + offset
                
                # Simulamos momios típicos (-110 para ambos lados)
                odds = -110
                decimal_odds = self.american_to_decimal(odds)
                
                # Over tiene valor si predicción > line
                is_over = offset < 0
                
                win_prob = self.calculate_win_probability(pred_pts, line, over=is_over)
                ev = self.calculate_ev(win_prob, decimal_odds)
                kelly = self.calculate_kelly(win_prob, decimal_odds)
                implied_prob = 0.524  # Prob implícita de -110
                
                results.append({
                    'player': player,
                    'prediction': pred_pts,
                    'line': line,
                    'type': 'OVER' if is_over else 'UNDER',
                    'odds': odds,
                    'decimal_odds': round(decimal_odds, 3),
                    'bookmaker': 'SIMULADO',
                    'win_prob': round(win_prob, 3),
                    'implied_prob': implied_prob,
                    'edge': round(win_prob - implied_prob, 3),
                    'ev_per_100': round(ev, 2),
                    'kelly_pct': round(kelly * 100, 2)
                })
        
        return pd.DataFrame(results).sort_values('ev_per_100', ascending=False)
    
    def find_value_bets(self, min_ev: float = 5.0, min_edge: float = 0.05) -> pd.DataFrame:
        """
        Encuentra apuestas con valor positivo.
        
        Args:
            min_ev: EV mínimo por $100 apostados
            min_edge: Ventaja mínima sobre la casa (5% = 0.05)
        """
        # Aquí cargarías predicciones reales del modelo
        # Por ahora, placeholder
        logger.warning("Implementar carga de predicciones desde predict.py")
        return pd.DataFrame()


def print_analysis(df: pd.DataFrame):
    """Imprime análisis de forma legible"""
    if df.empty:
        print("No se encontraron oportunidades.")
        return
    
    print("\n" + "="*80)
    print("🎯 ANÁLISIS DE VALOR ESPERADO (EV)")
    print("="*80)
    
    # Filtrar solo EV positivo
    value_bets = df[df['ev_per_100'] > 0]
    
    if value_bets.empty:
        print("\n⚠️  No hay apuestas con EV positivo en este momento.")
        print("\nTodas las oportunidades analizadas:")
        print(df[['player', 'line', 'type', 'odds', 'win_prob', 'ev_per_100']].head(10))
        return
    
    print(f"\n✅ Encontradas {len(value_bets)} apuestas con valor positivo:\n")
    
    for _, row in value_bets.head(10).iterrows():
        emoji = "🔥" if row['ev_per_100'] > 10 else "✅" if row['ev_per_100'] > 5 else "👀"
        print(f"{emoji} {row['player']}")
        print(f"   Predicción: {row['prediction']:.1f} pts | Línea: {row['line']} {row['type']}")
        print(f"   Momio: {row['odds']:+d} ({row['bookmaker']})")
        print(f"   P(Ganar): {row['win_prob']:.1%} vs Implícita: {row['implied_prob']:.1%}")
        print(f"   Edge: {row['edge']:+.1%} | EV: ${row['ev_per_100']:+.2f}/100 | Kelly: {row['kelly_pct']:.1f}%")
        print()


if __name__ == "__main__":
    calc = EVCalculator()
    
    # Ejemplo de uso con predicciones manuales
    # (En producción, esto vendría de predict.py)
    sample_predictions = {
        "LeBron James": 26.5,
        "Stephen Curry": 28.0,
        "Luka Doncic": 32.0,
        "Jayson Tatum": 27.5,
        "Giannis Antetokounmpo": 30.0
    }
    
    print("Analizando oportunidades de valor...")
    results = calc.analyze_player_props(sample_predictions)
    
    print_analysis(results)
    
    # Guardar resultados
    if not results.empty:
        output_path = calc.output_dir / "ev_analysis.parquet"
        results.to_parquet(output_path, index=False)
        logger.info(f"Análisis guardado en: {output_path}")