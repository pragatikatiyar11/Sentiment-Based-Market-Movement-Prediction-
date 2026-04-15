# Phase 2: Data Ingestion and Alignment - Quick Start

## Overview

Phase 2 builds the complete data pipeline that feeds into Phase 3. It orchestrates:
1. **Market Data Collection** — 3 years of OHLCV from Yahoo Finance
2. **Sentiment Text Collection** — News headlines for each stock
3. **Label Creation** — Binary price direction labels (UP/DOWN)
4. **Data Alignment** — Join market data + sentiment by date
5. **Validation** — Quality checks and dataset export

**Output**: `backend/data/processed/unified_training_dataset.csv`

---

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
cd backend
python pipeline.py
```

This runs all steps end-to-end and produces the unified training dataset.

### Option 2: Run Individual Steps

```bash
# Step 1: Collect market data and labels
python data_collector.py

# Step 2: Collect sentiment texts (news headlines)
python sentiment_collector.py

# Step 3: Join and validate
python pipeline.py
```

---

## Output Files

After successful completion, check:

```
backend/data/processed/
├── training_dataset.csv           # Market OHLCV + labels (from data_collector.py)
├── sentiment_texts.csv             # Aggregated headlines (from sentiment_collector.py)
└── unified_training_dataset.csv   # Final merged dataset (from pipeline.py)
```

---

## Dataset Schema

The unified dataset has columns:

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Stock ticker (AAPL, MSFT, etc.) |
| `date` | str | YYYY-MM-DD |
| `open` | float | Opening price |
| `high` | float | Daily high |
| `low` | float | Daily low |
| `close` | float | Closing price (prediction feature) |
| `volume` | float | Trading volume |
| `label` | int | 1=UP, 0=DOWN (next-day direction) |
| `aggregated_text` | str | Concatenated news headlines |
| `headline_count` | int | Number of headlines on that date |

---

## Validation Checks

The pipeline runs automatic validation:

- ✓ **Schema**: All required columns present
- ✓ **Nulls**: No NaN in critical columns (close, label)
- ✓ **Label Balance**: Reasonable UP/DOWN split (not all one class)
- ✓ **Duplicates**: No duplicate (symbol, date) pairs
- ✓ **Coverage**: All 10 symbols present
- ✓ **Sentiment Coverage**: % of rows with headlines

If any check fails, review the logs and consider:
- yfinance API issues → retry or use cached data
- Missing stock data → drop stock temporarily
- Headline collection → ensure NEWSAPI_KEY or use mock data

---

## Customization

Edit `backend/config.py` to change:

- `STOCK_UNIVERSE` — List of symbols to fetch
- `DATA_SOURCES` — Market data lookback window (3 years default)
- `SENTIMENT_CONFIG` — Aggregation windows and scoring

---

## Next: Phase 3

Once this dataset is ready, move to Phase 3:

```bash
python backend/ml_model/sentiment_extractor.py
```

This will add finance sentiment features and rolling aggregates.

---

## Troubleshooting

### "No data returned for SYMBOL"
- yfinance API may be temporarily unavailable
- Symbol may not be valid (check STOCK_UNIVERSE in config.py)
- Solution: Retry or remove from universe

### "No headlines collected"
- Using mock data by default (safe for MVP)
- To use real NewsAPI: Set `NEWSAPI_KEY` environment variable
- Solution: Proceed with market data only; Phase 3 handles empty text

### Import errors
- Ensure requirements.txt installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

---

## Performance Notes

- First run: ~2-5 minutes (depends on yfinance API speed)
- Subsequent runs: ~1 minute (if using cached data)
- Memory: ~500MB for 3 years × 10 stocks
- Storage: Final CSV ~50MB

---

**Status**: Phase 2 complete when `unified_training_dataset.csv` exists and all validation checks pass.
