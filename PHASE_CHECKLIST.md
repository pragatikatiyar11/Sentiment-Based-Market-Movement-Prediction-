# Sentiment-Based Market Movement Prediction - Phase Checklist

Use this checklist to track end-to-end progress for the backend ML pipeline build.

- [x] Phase 1 (Week 1): Problem Framing and Setup
- [x] Phase 2 (Week 1-2): Data Ingestion and Alignment
- [ ] Phase 3 (Week 2-3): Finance Sentiment Layer (FinBERT-class)
- [ ] Phase 4 (Week 3-4): Prediction Model and Feature Engineering
- [ ] Phase 5 (Week 4-5): Validation and Backtesting
- [ ] Phase 6 (Week 5-6): API Integration and Reliability
- [ ] Phase 7 (End of Week 6): Finalization and Presentation Packaging

## Optional Tracking Notes

- [x] Phase 1 deliverables complete
- [x] Phase 2 deliverables complete
- [ ] Phase 3 deliverables complete
- [ ] Phase 4 deliverables complete
- [ ] Phase 5 deliverables complete
- [ ] Phase 6 deliverables complete
- [ ] Phase 7 deliverables complete

## Week-Wise Checklist

### Week 1

- [x] Finalize prediction objective and label definition
- [x] Confirm stock universe and data source list
- [x] Set up environment and project conventions
- [x] Define API input/output contract

### Week 2

- [x] Build market data ingestion pipeline
- [x] Build text/news sentiment ingestion pipeline
- [x] Align timestamps and generate joined training dataset
- [x] Validate dataset quality and split strategy

### Week 3

- [ ] Integrate finance sentiment model (FinBERT-class)
- [ ] Generate sentiment features and rolling aggregates
- [ ] Engineer baseline technical indicators
- [ ] Start baseline model training

### Week 4

- [ ] Compare baseline models and pick best candidate
- [ ] Replace heuristic predictor with model inference
- [ ] Tune feature set based on validation performance

### Week 5

- [ ] Run out-of-sample evaluation
- [ ] Implement walk-forward backtesting
- [ ] Report return/drawdown and model stability

### Week 6

- [ ] Harden API validation and error handling
- [ ] Add logging and reproducible model loading
- [ ] Freeze artifacts and prepare final presentation summary
