# 🏀 NBA Player Points Prediction System

Sistema de predicción de puntos de jugadores NBA para identificar oportunidades de valor en apuestas deportivas.

## 📋 Descripción

Este proyecto utiliza Machine Learning (XGBoost) para predecir cuántos puntos anotará un jugador en un partido, comparando las predicciones con las líneas de las casas de apuestas para encontrar valor esperado positivo.

El modelo aprende patrones de:
- Rendimiento histórico del jugador (rolling stats)
- Contexto del partido (Vegas spread, O/U, home/away)
- Defensa del rival por posición
- Ritmo de juego (pace)

## 🏗️ Arquitectura

```
├── src/
│   ├── ingestion/              # Obtención de datos
│   │   ├── nba_stats.py        # API oficial NBA (BoxScoreTraditionalV3)
│   │   ├── odds_api.py         # The Odds API (momios en vivo)
│   │   ├── historical_odds.py  # Procesamiento de odds históricas (CSV)
│   │   └── backfill.py         # Carga histórica de datos
│   │
│   ├── processing/             # Transformación de datos
│   │   ├── features.py         # Features base (rolling stats, momentum)
│   │   ├── market_features.py  # Features de mercado (Vegas lines)
│   │   ├── team_stats.py       # Pace y stats de equipo
│   │   ├── position_defense.py # Defensa por posición del rival
│   │   └── defense.py          # Stats defensivas generales
│   │
│   ├── modeling/               # Machine Learning
│   │   ├── train.py            # Entrenamiento XGBoost
│   │   ├── optimize.py         # Optimización con Optuna (CV temporal)
│   │   ├── test_model.py       # Análisis de errores del modelo
│   │   └── predict.py          # Predicciones
│   │
│   └── strategy/               # Lógica de apuestas
│       ├── ev_calculator.py    # Expected Value calculator
│       └── bankroll.py         # Gestión de bankroll
│
├── data/
│   ├── raw/                    # Datos crudos (NBA API, Odds JSON)
│   ├── processed/              # Features procesadas (parquet)
│   │   ├── features/           # player_features_vX.parquet
│   │   └── odds_history_clean.parquet
│   ├── external/               # Datos externos (CSV odds históricas)
│   └── predictions/            # Predicciones generadas
│
├── notebooks/                  # Análisis exploratorio
└── logs/                       # Logs y gráficos de análisis
```

## 🔄 Pipeline de Datos

```
1. INGESTA                    2. PROCESAMIENTO                3. MODELADO
┌────────────────┐           ┌─────────────────────┐         ┌──────────────┐
│ NBA API        │──────────▶│ Rolling Stats       │────────▶│ XGBoost      │
│ (Box Scores)   │           │ (last_3, 5, 10)     │         │ Regressor    │
├────────────────┤           ├─────────────────────┤         │              │
│ Historical CSV │──────────▶│ Market Features     │────────▶│ MAE: 4.11    │
│ (Odds 2007-23) │           │ (spread, O/U)       │         │              │
├────────────────┤           ├─────────────────────┤         └──────────────┘
│ Odds API       │──────────▶│ Position Defense    │
│ (Live)         │           │ (opp_pts_to_pos)    │
└────────────────┘           └─────────────────────┘
```

## 📊 Features del Modelo

### Fase 1: Features Base del Jugador
- **Rolling averages**: `pts_last_3`, `pts_last_5`, `pts_last_10`
- **Variabilidad**: `pts_std_10`, `reb_std_10` (desviación estándar)
- **Techo/Suelo**: `pts_max_10`, `pts_min_10`
- **Eficiencia**: `ppm_last_10` (puntos por minuto histórico)
- **Momentum**: `pts_momentum` (last_3 - last_10, detecta rachas)
- **Contexto**: `is_home`, `rest_days`, `is_b2b` (back-to-back)

### Fase 1: Features de Mercado (Vegas)
- **`vegas_spread`**: Handicap del partido (favorito vs underdog)
- **`vegas_total`**: Over/Under total del partido
- **`implied_team_score`**: Puntos esperados del equipo
- **`is_favorite`**: 1 si es favorito, 0 si underdog
- **`expected_blowout`**: 1 si |spread| > 10 (riesgo garbage time)

### Fase 2: Features de Defensa por Posición
- **`opp_pts_to_pos_last_10`**: Puntos que el rival permite a la posición del jugador
- **`opp_def_rating_last_5`**: Rating defensivo del rival (últimos 5 juegos)
- **`opp_pace_last_10`**: Ritmo de juego del rival

### Variables Excluidas (Data Leakage)
- `min` - Minutos jugados en el partido actual
- `plusminuspoints` - +/- del partido actual
- `pts`, `reb`, `ast` del partido actual
- Cualquier stat que solo conoces después del partido

## 🚀 Uso

### 1. Instalar dependencias
```bash
uv sync
```

### 2. Obtener datos históricos (NBA Stats)
```bash
# Descargar 3 meses de datos (Oct-Dic 2022)
python -m src.ingestion.backfill
```

### 3. Procesar odds históricas (si tienes CSV)
```bash
python -m src.ingestion.historical_odds
```

### 4. Generar features (ejecutar en orden)
```bash
python -m src.processing.features
python -m src.processing.team_stats
python -m src.processing.market_features
python -m src.processing.position_defense
```

### 5. Optimizar hiperparámetros
```bash
python -m src.modeling.optimize
```

### 6. Entrenar modelo
```bash
python -m src.modeling.train
```

### 7. Analizar errores
```bash
python -m src.modeling.test_model
```

## 📈 Métricas Actuales

| Métrica | Valor |
|---------|-------|
| **MAE General** | 4.11 puntos |
| MAE Guards (G) | 3.84 puntos |
| MAE Forwards (F) | 5.05 puntos |
| MAE Centers (C) | 4.87 puntos |
| Train/Test Split | 80/20 temporal |
| Registros de entrenamiento | ~12,000 |

### Análisis de Errores por Contexto
| Contexto | MAE |
|----------|-----|
| Partido cerrado | 4.03 |
| Expected blowout | 5.29 (+31%) |
| Descansado | 3.99 |
| Back-to-back | 5.06 (+27%) |

### Top Features por Importancia
1. `vegas_spread` - Contexto del partido
2. `min_last_10` - Minutos históricos
3. `min_max_10` - Techo de minutos
4. `implied_team_score` - Puntos esperados del equipo
5. `opp_pace_last_10` - Ritmo del rival

## 🔮 Próximos Pasos (Roadmap)

### Fase 3A: Rol del Jugador
- [ ] `player_role.py` - Clasificación por tier (Estrella/Titular/Rotación/Banca)
- [ ] Features: `usage_pct`, `min_share`, `pts_share`

### Fase 3B: Predicción de Minutos
- [ ] `train_minutes.py` - Modelo auxiliar para predecir minutos
- [ ] Feature: `pred_minutes` para usar en predicción de puntos

### Fase 4: Pipeline de Producción
- [ ] `live_features.py` - Features para juegos de hoy
- [ ] `predict.py` - Pipeline completo de predicción
- [ ] `recommendations.py` - Output de recomendaciones OVER/UNDER

### Completados ✅
- [x] Features de mercado (Vegas spread, O/U)
- [x] Defensa por posición del rival
- [x] Momentum (racha caliente/fría)
- [x] Back-to-back detection
- [x] Análisis de errores del modelo

## ⚙️ Tecnologías

- **Python 3.12+**
- **XGBoost** - Modelo de predicción (MAE 4.11)
- **Optuna** - Optimización de hiperparámetros (CV temporal)
- **Pandas/NumPy/PyArrow** - Procesamiento de datos
- **nba_api** - Datos oficiales NBA (BoxScoreTraditionalV3)
- **The Odds API** - Momios de casas de apuestas
- **Matplotlib/Seaborn** - Visualización de errores

## 📝 Notas

Este proyecto es para fines educativos y de análisis. Las apuestas deportivas conllevan riesgo financiero.

## 📄 Licencia

MIT
