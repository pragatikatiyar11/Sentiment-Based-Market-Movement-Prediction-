# Phase 4: Prediction Model & Feature Engineering - Quick Start

## Overview

Phase 4 trains the prediction model using features from all previous phases. It:
1. Engineers features (market + technical + sentiment)
2. Splits data into train/validation/test
3. Trains multiple candidate models
4. Selects and evaluates the best model
5. Saves serialized model for Phase 6 API deployment

**Input**: `backend/data/processed/sentiment_augmented_dataset.csv` (Phase 3 output)  
**Output**: Trained model + scaler files in `backend/ml_model/trained_models/`

---

## Quick Start

### Run Phase 4 Training

```bash
cd backend
python ml_model/train.py
```

This will:
1. Load Phase 3 sentiment-augmented dataset
2. Engineer market, technical, and sentiment features
3. Train logistic regression and random forest models
4. Evaluate on test set
5. Save best model

### Expected Output

```
✓ Loaded 7500 rows
✓ Engineered 18 features
✓ Train: 5250 | Val: 1050 | Test: 1200
✓ Best model: random_forest (val accuracy: 0.563)
✓ Test accuracy: 0.552
✓ Model saved: backend/ml_model/trained_models/random_forest_20260416_123456.pkl
```

---

## Features Generated

Phase 4 combines features from all previous phases:

### Market Features
- `close_price` — Closing price at time t
- `volume` — Trading volume

### Technical Indicators
- `sma_5`, `sma_20` — Simple moving averages
- `volatility_5`, `volatility_20` — Price volatility (rolling std)
- `price_change` — Log return

### Sentiment Features (from Phase 3)
- `score` — Sentiment score [0, 1]
- `sentiment_ma_1`, `sentiment_ma_5`, `sentiment_ma_20` — Rolling sentiment aggregates
- `headline_count` — Number of headlines on that date

### Lag Features (Momentum)
- `price_lag_1`, `price_lag_5` — Past prices

**Total**: 18 features

---

## Candidate Models

Phase 4 trains and compares:

| Model | Type | Speed | Interpretability |
|-------|------|-------|------------------|
| **Logistic Regression** | Linear | Fast | High |
| **Random Forest** | Tree ensemble | Medium | Medium |

Selection: Best validation accuracy wins. Tie-break on inference speed (for Phase 6 API).

---

## Data Splitting Strategy

**Time-aware splits** (prevent data leakage):

- **Train**: First 65% of historical data
- **Validation**: Next 20% (for hyperparameter tuning)
- **Test**: Last 15% (final holdout evaluation)

This mimics real-world prediction: train on past, predict on future.

---

## Training Configuration

Edit `backend/config.py` to customize:

```python
TRAINING_CONFIG = {
    "test_split_ratio": 0.2,              # 20% test data
    "validation_split_ratio": 0.15,       # 15% validation
    "walk_forward_window": 252,           # 1 year training window
    "walk_forward_step": 21,              # 1 month slide
    "random_seed": 42,
}

CANDIDATE_MODELS = [
    "logistic_regression",
    "random_forest",
    # "xgboost",      # Phase 4+ extension
    # "lightgbm",     # Phase 4+ extension
]
```

---

## Model Output

Trained model files are saved to `backend/ml_model/trained_models/`:

```
trained_models/
├── random_forest_20260416_123456.pkl
├── scaler_20260416_123456.pkl          # Feature normalization (must use same scaler!)
└── model_metadata_20260416_123456.json # (Optional) Training date, features, accuracy
```

---

## Expected Accuracy

**Baseline (random)**: 50% (coin flip for UP/DOWN)  
**Phase 4 target**: > 55% (5% above baseline)  
**Phase 5 goal**: > 55% with drawdown < 15%

---

## Customization

To add features:
1. Edit `engineer_features()` in `train.py`
2. Add new column to features_dict
3. Rerun training

To add models:
1. Import model class (e.g., `from sklearn.ensemble import GradientBoostingClassifier`)
2. Add to `train_candidate_models()` function
3. Add model name to `CANDIDATE_MODELS` in config.py

---

## Troubleshooting

### "Dataset not found"
- Run Phase 3 first: `python backend/ml_model/sentiment_extractor.py`

### "Poor accuracy (< 50%)"
- Check feature engineering: Are features meaningful?
- Validate labels: Are UP/DOWN balanced?
- Try different models or hyperparameters

### Memory error
- Reduce dataset size (fewer symbols or years)
- Use `n_jobs=1` in RandomForest instead of `-1`

---

## Next: Phase 5

Once model is trained:

```bash
python backend/ml_model/evaluate.py
```

This runs walk-forward backtesting to measure real-world performance.

---

**Status**: Phase 4 complete when:
- [ ] Model trained and saved
- [ ] Test accuracy > 50% (ideally > 55%)
- [ ] Model files exist in `trained_models/`
