"""
Análisis de Errores del Modelo

Este script analiza los errores de predicción para identificar:
1. Casos con errores extremos (>15 pts)
2. Patrones por posición, equipo, contexto
3. Oportunidades de mejora del modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error
from src.config import PROCESSED_DIR, BASE_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    def __init__(self):
        self.features_path = PROCESSED_DIR / "features" / "player_features_v2_position.parquet"
        self.model_path = BASE_DIR / "src" / "modeling" / "model_store" / "xgb_points_v1.pkl"
    
    def load_data_and_model(self):
        """Carga datos y modelo entrenado"""
        df = pd.read_parquet(self.features_path)
        df = df.dropna(subset=['pts_last_5', 'pts'])
        df = df.sort_values(by='game_date').reset_index(drop=True)
        
        model = joblib.load(self.model_path)
        
        # Split igual que en train.py
        split_idx = int(len(df) * 0.8)
        df_test = df.iloc[split_idx:].copy()
        
        return df_test, model
    
    def get_feature_columns(self, df):
        """Obtiene las mismas features que usa el modelo"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        forbidden_cols = [
            'player_id', 'game_id', 'team_id', 'teamid', 'team_id_opp',
            'min', 'plusminuspoints', 'ppm',
            'pts', 'reb', 'ast', 'stl', 'blk', 'tov',
            'dreb', 'oreb',
            'fieldgoalsmade', 'fieldgoalsattempted', 'fieldgoalspercentage',
            'threepointersmade', 'threepointersattempted', 'threepointerspercentage',
            'freethrowsmade', 'freethrowsattempted', 'freethrowspercentage',
            'foulspersonal', 'possessions', 'team_pace',
            'points_allowed', 'points_allowed_actual', 'opp_pace_actual',
        ]
        
        features = [
            col for col in numeric_cols
            if (('last' in col or 'std' in col or 'max_10' in col or 'min_10' in col)
                and col not in forbidden_cols)
        ]
        
        for cf in ['is_home', 'rest_days']:
            if cf in numeric_cols and cf not in features:
                features.append(cf)
        
        for tf in ['pace_last_10', 'opp_pace_last_10', 'opp_pts_allowed_last_10']:
            if tf in numeric_cols and tf not in features:
                features.append(tf)
        
        for mf in ['vegas_total', 'vegas_spread', 'is_favorite', 'expected_blowout', 'implied_team_score']:
            if mf in numeric_cols and mf not in features:
                features.append(mf)
        
        for pf in ['opp_pts_to_pos_last_10', 'opp_def_rating_last_5']:
            if pf in numeric_cols and pf not in features:
                features.append(pf)
        
        return features
    
    def analyze(self):
        """Ejecuta el análisis completo"""
        df_test, model = self.load_data_and_model()
        features = self.get_feature_columns(df_test)
        
        # Predicciones
        X_test = df_test[features]
        df_test['pred_pts'] = model.predict(X_test)
        df_test['error'] = df_test['pts'] - df_test['pred_pts']
        df_test['abs_error'] = df_test['error'].abs()
        
        logger.info(f"=== ANÁLISIS DE ERRORES ===")
        logger.info(f"Test set: {len(df_test)} registros")
        logger.info(f"MAE: {df_test['abs_error'].mean():.2f} puntos")
        
        # 1. Distribución de errores
        self._plot_error_distribution(df_test)
        
        # 2. Errores extremos
        self._analyze_extreme_errors(df_test)
        
        # 3. Errores por posición
        self._analyze_by_position(df_test)
        
        # 4. Errores por rango de puntos reales
        self._analyze_by_pts_range(df_test)
        
        # 5. Errores por contexto (B2B, Home/Away)
        self._analyze_by_context(df_test)
        
        # 6. Jugadores más difíciles de predecir
        self._analyze_hardest_players(df_test)
        
        return df_test
    
    def _plot_error_distribution(self, df):
        """Histograma de distribución de errores"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error distribution
        axes[0].hist(df['error'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', label='Error = 0')
        axes[0].set_xlabel('Error (Real - Predicción)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Errores')
        axes[0].legend()
        
        # Predicción vs Realidad
        axes[1].scatter(df['pts'], df['pred_pts'], alpha=0.3, s=10)
        axes[1].plot([0, 50], [0, 50], 'r--', label='Predicción perfecta')
        axes[1].set_xlabel('Puntos Reales')
        axes[1].set_ylabel('Puntos Predichos')
        axes[1].set_title('Predicción vs Realidad')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(BASE_DIR / 'logs' / 'error_distribution.png', dpi=150)
        plt.show()
        logger.info("📊 Gráfico guardado: logs/error_distribution.png")
    
    def _analyze_extreme_errors(self, df, threshold=15):
        """Analiza casos con errores mayores a threshold"""
        extreme = df[df['abs_error'] > threshold].copy()
        
        print(f"\n{'='*60}")
        print(f"ERRORES EXTREMOS (|error| > {threshold} pts): {len(extreme)} casos ({len(extreme)/len(df)*100:.1f}%)")
        print('='*60)
        
        if len(extreme) == 0:
            print("No hay errores extremos.")
            return
        
        # Mostrar ejemplos
        cols_to_show = ['game_date', 'player_name', 'position', 'pts', 'pred_pts', 'error', 
                        'min_last_10', 'pts_last_10', 'vegas_spread']
        available_cols = [c for c in cols_to_show if c in extreme.columns]
        
        print("\nTop 10 errores más grandes:")
        print(extreme.nlargest(10, 'abs_error')[available_cols].to_string())
        
        # Patrones en errores extremos
        print(f"\n--- Patrones en Errores Extremos ---")
        print(f"Subestimaciones (modelo predijo menos): {(extreme['error'] > 0).sum()}")
        print(f"Sobreestimaciones (modelo predijo más): {(extreme['error'] < 0).sum()}")
        
        if 'position' in extreme.columns:
            print(f"\nPor posición:")
            print(extreme.groupby('position')['abs_error'].agg(['count', 'mean']))
    
    def _analyze_by_position(self, df):
        """MAE por posición"""
        if 'position_base' not in df.columns:
            return
        
        print(f"\n{'='*60}")
        print("MAE POR POSICIÓN")
        print('='*60)
        
        pos_stats = df.groupby('position_base').agg({
            'abs_error': ['mean', 'std', 'count'],
            'pts': 'mean'
        }).round(2)
        pos_stats.columns = ['MAE', 'STD', 'Count', 'Avg_Pts']
        print(pos_stats.sort_values('MAE'))
    
    def _analyze_by_pts_range(self, df):
        """MAE por rango de puntos reales"""
        print(f"\n{'='*60}")
        print("MAE POR RANGO DE PUNTOS REALES")
        print('='*60)
        
        df['pts_range'] = pd.cut(df['pts'], 
                                  bins=[0, 5, 10, 15, 20, 25, 100],
                                  labels=['0-5', '6-10', '11-15', '16-20', '21-25', '25+'])
        
        range_stats = df.groupby('pts_range').agg({
            'abs_error': ['mean', 'count'],
            'error': 'mean'  # Sesgo
        }).round(2)
        range_stats.columns = ['MAE', 'Count', 'Bias']
        print(range_stats)
        
        # Interpretación
        print("\n💡 Interpretación:")
        print("   - Bias positivo = Modelo subestima (predice menos de lo real)")
        print("   - Bias negativo = Modelo sobreestima (predice más de lo real)")
    
    def _analyze_by_context(self, df):
        """MAE por contexto del juego"""
        print(f"\n{'='*60}")
        print("MAE POR CONTEXTO")
        print('='*60)
        
        # Home vs Away
        if 'is_home' in df.columns:
            home_stats = df.groupby('is_home')['abs_error'].mean()
            home_val = home_stats.get(1, None)
            away_val = home_stats.get(0, None)
            print(f"Local (is_home=1): MAE = {home_val:.2f}" if home_val else "Local: N/A")
            print(f"Visita (is_home=0): MAE = {away_val:.2f}" if away_val else "Visita: N/A")
        
        # Back to Back
        if 'is_b2b' in df.columns:
            b2b_stats = df.groupby('is_b2b')['abs_error'].mean()
            rest_val = b2b_stats.get(0, None)
            b2b_val = b2b_stats.get(1, None)
            print(f"\nDescansado (is_b2b=0): MAE = {rest_val:.2f}" if rest_val else "\nDescansado: N/A")
            print(f"Back-to-back (is_b2b=1): MAE = {b2b_val:.2f}" if b2b_val else "B2B: N/A")
        
        # Blowouts
        if 'expected_blowout' in df.columns:
            blowout_stats = df.groupby('expected_blowout')['abs_error'].mean()
            close_val = blowout_stats.get(0, None)
            blowout_val = blowout_stats.get(1, None)
            print(f"\nPartido cerrado: MAE = {close_val:.2f}" if close_val else "\nPartido cerrado: N/A")
            print(f"Expected blowout: MAE = {blowout_val:.2f}" if blowout_val else "Blowout: N/A")
            
    def _analyze_hardest_players(self, df, top_n=10):
        """Jugadores más difíciles de predecir"""
        if 'player_name' not in df.columns:
            if 'player_id' in df.columns:
                player_col = 'player_id'
            else:
                return
        else:
            player_col = 'player_name'
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} JUGADORES MÁS DIFÍCILES DE PREDECIR")
        print('='*60)
        
        player_stats = df.groupby(player_col).agg({
            'abs_error': ['mean', 'std', 'count'],
            'pts': 'mean'
        })
        player_stats.columns = ['MAE', 'STD', 'Games', 'Avg_Pts']
        player_stats = player_stats[player_stats['Games'] >= 5]  # Mínimo 5 juegos
        
        print("\nMayor MAE (inconsistentes):")
        print(player_stats.nlargest(top_n, 'MAE')[['MAE', 'STD', 'Games', 'Avg_Pts']].to_string())
        
        print("\nMenor MAE (más predecibles):")
        print(player_stats.nsmallest(top_n, 'MAE')[['MAE', 'STD', 'Games', 'Avg_Pts']].to_string())


if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    df_results = analyzer.analyze()
    
    # Guardar resultados para análisis posterior
    output_path = PROCESSED_DIR / "analysis" / "error_analysis.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(output_path)
    logger.info(f"\n✅ Resultados guardados en: {output_path}")