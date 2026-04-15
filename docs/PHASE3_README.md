# Phase 3: Finance Sentiment Layer - Quick Start

## Overview

Phase 3 augments the Phase 2 dataset with sentiment features extracted from news headlines. It:
1. Loads a finance-tuned sentiment model (FinBERT-class)
2. Extracts sentiment scores (0-1) from aggregated headlines
3. Creates rolling sentiment aggregates (1-day, 5-day, 20-day MA)
4. Outputs sentiment-enriched dataset for Phase 4

**Input**: `backend/data/processed/unified_training_dataset.csv` (Phase 2 output)  
**Output**: `backend/data/processed/sentiment_augmented_dataset.csv`

---

## Quick Start

### Run Phase 3 End-to-End

```bash
cd backend
python ml_model/sentiment_extractor.py
```

This will:
1. Load Phase 2 dataset
2. Extract sentiment from headlines
3. Create rolling aggregates
4. Save augmented dataset

### Expected Output

```
✓ Loaded 7500 rows from unified_training_dataset.csv
✓ Loading sentiment model (ProsusAI/finBERT)
✓ Extracted sentiment features for 7500 rows
✓ Created 3 rolling sentiment features
✓ Saved to sentiment_augmented_dataset.csv
```

---

## Sentiment Features Generated

After Phase 3, the dataset has new columns:

| Column | Type | Description |
|--------|------|-------------|
| `score` | float [0, 1] | Sentiment score (0=negative, 1=positive) |
| `label` | str | POSITIVE / NEGATIVE / NEUTRAL |
| `confidence` | float [0, 1] | Model confidence in prediction |
| `sentiment_ma_1` | float | 1-day rolling avg sentiment |
| `sentiment_ma_5` | float | 5-day rolling avg sentiment |
| `sentiment_ma_20` | float | 20-day rolling avg sentiment |

---

## Model Selection

**Phase 3 MVP**: Uses `distilbert-base-uncased-finetuned-sst-2-english` (fast, general-purpose)

**Phase 7 Production**: Upgrade to `ProsusAI/finBERT-tone` for finance-specific sentiment

To override, edit `backend/config.py`:
```python
SENTIMENT_CONFIG = {
    "model_class": "FinBERT",  # or any HuggingFace model ID
}
```

---

## Customization

Edit `backend/config.py` to modify:

```python
SENTIMENT_CONFIG = {
    "model_class": "FinBERT",
    "aggregation_windows": [1, 5, 20],  # Rolling window sizes
    "score_normalization": "min_max",   # How to normalize scores
    "confidence_threshold": 0.5,        # Min confidence to include
}
```

---

## Performance

- **First run**: 2-5 minutes (model download + inference on all rows)
- **Subsequent runs**: 30 seconds (cached model)
- **Memory**: ~2GB (for transformer model + data)
- **Output size**: ~80MB CSV

---

## Quality Notes

- **Empty headlines**: Treated as neutral (0.5 score)
- **Very long text**: Truncated to 512 tokens (BERT limit)
- **Rolling aggregates**: Only use available data at beginning (min_periods=1)

---

## Next: Phase 4

Once sentiment features are ready:

```bash
python backend/ml_model/train.py
```

This will train the prediction model using market + sentiment features.

---

**Status**: Phase 3 complete when `sentiment_augmented_dataset.csv` exists with sentiment columns and rolling aggregates.
