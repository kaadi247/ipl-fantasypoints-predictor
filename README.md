# IPL Fantasy Points Predictor

A machine learning project that predicts per-match Dream11 fantasy points for IPL players, trained on ball-by-ball delivery data from 17 seasons (2007–2024).

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (predict training mean) | 21.27 | — | — |
| XGBoost — L1 objective, tuned | 19.28 | — | — |
| **LightGBM — L1 objective, tuned** | **19.25** | **27.53** | **0.033** |

> **Note on R²:** A low R² (0.033) is expected for T20 cricket — individual match performance is highly stochastic (pitch conditions, match situation, injury), and the same player can score 0 pts or 100 pts in different matches. The meaningful signal is the **9.5% MAE reduction** over a strong mean-reversion baseline (21.27 → 19.25), achieved using only pre-match historical features with no leakage.

---

## Dataset

- **Source:** Cricsheet ball-by-ball IPL delivery data (`deliveries.csv`, `matches.csv`)
- **Coverage:** 17 seasons, 2007–2024 (1,095 matches, 260,920 deliveries)
- **Target variable:** Dream11 fantasy points per player per match, computed from raw deliveries
- **Train / holdout split:** 2007–2022 (18,440 player-match rows), 2023–2024 (3,067 rows). The split is strictly temporal so that all validation metrics reflect genuine out-of-sample forecasts with no leakage.

---

## Dream11 Scoring Logic

Fantasy points are calculated from the raw delivery data in `01_eda.ipynb` and validated against historical Dream11 records. The scoring rules implemented are:

**Batting**
- 0.5 pts per run
- 1 pt per four, 2 pts per six
- Milestone bonuses: +4 at 25 runs, +8 at 50, +8 at 75, **+16 at 100**
- −2 pts for a duck (dismissed for 0)

**Bowling**
- 25 pts per wicket
- 8 pts per LBW or bowled dismissal
- Bonus tiers: +4 for 3-wicket haul, +8 additional for 4-wicket haul, **+16 additional for 5-wicket haul**
- 12 pts per maiden over

**Fielding**
- 8 pts per catch (including caught-and-bowled, credited to the bowler); 12 pts per stumping; 6 pts per run-out

Computed points are stored in `data/fantasy_points.csv` and used as the prediction target throughout the modelling pipeline.

---

## Methodology

### Feature Engineering (`02_features.ipynb`)

All features are computed as expanding-window or lag statistics so that only data preceding each match is used at prediction time. The 12 features used by the model are:

| Feature | Description |
|---|---|
| `career_avg` | Expanding mean of fantasy points across all prior appearances |
| `rolling_5_avg` | Rolling 5-match average (shifted by 1 to avoid leakage) |
| `current_season_avg` | Expanding within-season mean, reset each season |
| `career_std` | Expanding standard deviation of fantasy points (volatility signal) |
| `career_max` | Expanding career-best single-match score |
| `venue_avg` | Expanding mean at the specific venue |
| `home_away_avg` | Expanding mean split by home/away |
| `vs_opponent_avg` | Expanding mean against the specific opposition |
| `innings_avg` | Expanding mean split by batting innings (setter vs. chaser) |
| `avg_overs_bowled` | Expanding mean overs bowled per match (bowling role proxy) |
| `avg_balls_faced` | Expanding mean balls faced per match (batting role proxy) |
| `career_sr` | Expanding career strike rate |

NaN values for debut matches are imputed with `DEBUT_AVG = 27` (the approximate league mean). Bowling/batting proxy features are filled with 0 for non-bowlers/non-batters.

### Model Selection and Objective (`03_model.ipynb`)

The evaluation metric is MAE (Mean Absolute Error). Using an MSE objective caused tree-based models to bias predictions toward the mean and fail to capture the wide variance in T20 performances. Switching to an **L1 (absolute error) objective** aligns the loss function directly with the evaluation metric and improved holdout MAE.

Models benchmarked:
- Baseline (predict training mean)
- XGBoost — `objective='reg:absoluteerror'`
- **LightGBM — `objective='regression_l1'`** ← best result

Hyperparameters were tuned via `RandomizedSearchCV` (40 iterations, 3-fold CV) on the training fold only. Categorical variables (`venue`, `opponent`) are encoded as numerical proxy features (`venue_avg`, `vs_opponent_avg`) rather than raw labels, avoiding ordinal encoding while preserving contextual signal.

---

## Diagrams

### Feature Importance
![Feature Importance](diagrams/feature_importance.png)

### Predicted vs. Actual
![Predicted vs Actual](diagrams/predicted_vs_actual.png)

### Residual Distribution
![Residual Distribution](diagrams/residuals_distribution.png)

---

## Repository Structure

```
.
├── 01_eda.ipynb              # Data audit, Dream11 scoring computation, EDA
├── 02_features.ipynb         # Leakage-free feature engineering
├── 03_model.ipynb            # Model training, tuning, evaluation
├── data/
│   ├── deliveries.csv        # Raw ball-by-ball data (gitignored)
│   ├── matches.csv           # Match metadata (gitignored)
│   ├── fantasy_points.csv    # Computed targets (gitignored)
│   ├── featured_dataset.csv  # Final feature matrix (gitignored)
│   └── champion_model.pkl    # Serialised LightGBM model (gitignored)
├── diagrams/                 # Output plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order: `01_eda` → `02_features` → `03_model`.

---

## Resume Description

Built an end-to-end machine learning pipeline to predict IPL player Dream11 fantasy points from 17 seasons of ball-by-ball delivery data (2007–2024, 260K+ deliveries). Engineered a custom Dream11 scoring engine and 12 leakage-free features using expanding-window statistics (career averages, rolling form, venue/opponent/innings splits). Applied a strict temporal train/holdout split (2007–2022 / 2023–2024) to simulate real forecasting conditions. Switching the LightGBM objective from MSE to L1 loss — aligning training loss with the MAE evaluation metric — reduced holdout MAE by 9.5% over a mean-reversion baseline (21.27 → 19.25), with RMSE of 27.53 and R² of 0.033 on the holdout set. Low R² reflects the inherent stochasticity of T20 cricket rather than model deficiency.
