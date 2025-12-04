# 🏀 NBA Player Points Prediction System

Sistema de predicción de puntos de jugadores NBA para identificar oportunidades de valor en apuestas deportivas.

## 📋 Descripción

Este proyecto utiliza Machine Learning (XGBoost) para predecir cuántos puntos anotará un jugador en un partido, comparando las predicciones con las líneas de las casas de apuestas para encontrar valor esperado positivo.

## 🏗️ Arquitectura

```
├── src/
│   ├── ingestion/          # Obtención de datos
│   │   ├── nba_stats.py    # API oficial NBA (box scores)
│   │   ├── odds_api.py     # The Odds API (momios)
│   │   └── backfill.py     # Carga histórica de datos
│   │
│   ├── processing/         # Transformación de datos
│   │   ├── features.py     # Feature engineering
│   │   ├── cleaner.py      # Limpieza de datos
│   │   └── odds_processor.py
│   │
│   ├── modeling/           # Machine Learning
│   │   ├── train.py        # Entrenamiento XGBoost
│   │   ├── optimize.py     # Optimización con Optuna
│   │   └── predict.py      # Predicciones
│   │
│   └── strategy/           # Lógica de apuestas
│       ├── ev_calculator.py
│       └── bankroll.py
│
├── data/
│   ├── raw/                # Datos crudos (NBA API, Odds)
│   ├── processed/          # Features procesadas
│   └── predictions/        # Predicciones generadas
│
├── notebooks/              # Análisis exploratorio
└── tests/                  # Tests unitarios
```

## 🔄 Pipeline de Datos

```
1. INGESTA              2. PROCESAMIENTO           3. MODELADO
┌─────────────┐         ┌─────────────────┐        ┌─────────────┐
│ NBA API     │────────▶│ Rolling Stats   │───────▶│ XGBoost     │
│ (Box Scores)│         │ (last_5, last_10)│       │ Regressor   │
└─────────────┘         │ Contexto (H/A)  │        └─────────────┘
                        └─────────────────┘
```

## 📊 Features Actuales

### Variables Predictivas
- **Rolling averages**: `pts_last_5`, `pts_last_10`, `reb_last_X`, `ast_last_X`
- **Variabilidad**: `pts_std_10`, `reb_std_10` (desviación estándar)
- **Rangos**: `pts_max_10`, `pts_min_10`
- **Contexto**: `is_home`, `rest_days` (días de descanso)

### Variables Excluidas (Data Leakage)
- `min` - Minutos jugados en el partido actual
- `plusminuspoints` - +/- del partido actual
- Cualquier stat del partido que se está prediciendo

## 🚀 Uso

### 1. Instalar dependencias
```bash
uv sync
```

### 2. Obtener datos históricos
```bash
python -m src.ingestion.backfill
```

### 3. Generar features
```bash
python -m src.processing.features
```

### 4. Optimizar hiperparámetros (opcional)
```bash
python -m src.modeling.optimize
```

### 5. Entrenar modelo
```bash
python -m src.modeling.train
```

## 📈 Métricas Actuales

| Métrica | Valor |
|---------|-------|
| MAE (Error Promedio) | ~4-5 puntos |
| Train/Test Split | 80/20 temporal |

## 🔮 Próximos Pasos

- [ ] Agregar estadísticas defensivas del rival
- [ ] Incorporar pace (ritmo de juego) del oponente
- [ ] Detectar back-to-back games
- [ ] Integrar predicciones con momios en tiempo real
- [ ] Sistema de alertas para apuestas de valor

## ⚙️ Tecnologías

- **Python 3.12+**
- **XGBoost** - Modelo de predicción
- **Optuna** - Optimización de hiperparámetros
- **Pandas/NumPy** - Procesamiento de datos
- **nba_api** - Datos oficiales NBA
- **The Odds API** - Momios de casas de apuestas

## 📝 Notas

Este proyecto es para fines educativos y de análisis. Las apuestas deportivas conllevan riesgo financiero.
