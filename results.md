# Results

Every experiment gets a row in this table, and its prediction is written down before the run starts.

| Run | Model | What changed | Val PER | Val WER | Notes |
|-----|-------|--------------|---------|---------|-------|
| Run 1 | Pretrained GRU (official checkpoint) | none | XX.XX% | — | greedy CTC decoding, float32 inference on Kaggle T4, 1,426 validation trials |
| 2 | GRU, my training (official settings) | 600 batches, batch size 32 (speed test) | XX.XX% | — | float32, Kaggle T4, 1.41 s/batch including compile and validation; val PER fell from 121.6% to XX.XX% over 3 checks |
| 3 | GRU, my training (official settings) | 12,000 batches, batch size 32 | 13.43% | — | best val checkpoint (final check was still improving); Kaggle T4, float32, 3h26m, 1.03 s/batch; official 120k-batch, batch-64 run reports 10.1% |

## Training setup (all my runs)

- Kaggle T4 (one of two GPUs), float32 (`use_amp` false)
- Batch size 32 (official is 64; halved to avoid out-of-memory on a 15 GB T4)
- Cosine learning-rate schedule shortened to the number of batches; warmup is min(1000, 10% of batches)
- Official GRU architecture: 5 layers, 768 units, patch size 14, stride 4, day-specific input layers
- PER measured with greedy CTC decoding
- Rule: every model, including the transformer, uses the same batch size and number of batches

## Takeaways

- Reached 13.43% val PER with about 5% of the official run’s training samples (384k vs 7.68M), within 3.3 points of the official 10.1%.

## Models compared

| Part | Transformer | GRU (official) |
|------|------------:|---------------:|
| Day layers (shared design) | 11,819,520 | 11,819,520 |
| Projection | 2,752,896 | none |
| Encoder / GRU layers | 7,098,624 | 32,463,360 |
| Output | 15,785 | 31,529 |
| Initial hidden state | none | 768 |
| Total | 21,686,825 | 44,315,177 |

Transformer: d_model 384, 4 layers, 6 heads, feed-forward 1536. The transformer is about 49% of the GRU’s size; excluding the shared day layers, about 9.9M vs 32.5M.

## Known differences between the models

Weight decay (0.001) in the official optimizer is skipped only for parameters named gru.bias and out.bias, and the day layers get their own settings. After renaming the transformer’s output layer to out, its output bias is treated like the GRU’s. The encoder’s biases and LayerNorm weights are still decayed, which the GRU’s biases are not. Expected effect is small at this weight-decay value.
