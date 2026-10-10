# Results

Every experiment gets a row in this table, and its prediction is written down before the run starts.

| Run | Model | What changed | Val PER | Val WER | Notes |
|-----|-------|--------------|---------|---------|-------|
| Run 1 | Pretrained GRU (official checkpoint) | none | XX.XX% | — | greedy CTC decoding, float32 inference on Kaggle T4, 1,426 validation trials |
