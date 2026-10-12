# Results

Every experiment gets a row in this table, and its prediction is written down before the run starts.

| Run | Model | What changed | Val PER | Val WER | Notes |
|-----|-------|--------------|---------|---------|-------|
| 1 | Pretrained GRU (official checkpoint) | none | 10.19% | \ | greedy CTC decoding, float32 inference on Kaggle T4, 1,426 validation trials |
| 2 | GRU, my training (official settings) | 600 batches, batch size 32 (speed test) |  |  | float32, Kaggle T4, 1.41 s/batch including compile and validation; val PER fell over 3 checks |
| 3 | GRU, my training (official settings) | 12,000 batches, batch size 32 | 13.43% |  | best val checkpoint (final check was still improving); Kaggle T4, float32, 3h26m, 1.03 s/batch; official 120k-batch, batch-64 run reports 10.1% |
| 4 | Transformer (mine) | 4 layers, d_model 384, 6 heads, FF 1536; encoder lr 1e-3; torch.compile off; 12,000 batches, batch size 32 | 52.61% |  | Best checkpoint at batch 10,800 (final 52.63%), flattening as the cosine schedule ended. Train loss ended around 35-47 vs about 3-5 for the GRU, so it underfit. No prediction was recorded before this run. Differs from run 3 in learning rate (1e-3 vs 5e-3), compile off, and weight decay on encoder biases and LayerNorm. T4, float32, 87 min, 0.44 s/batch. |
| 5 | Transformer (mine) | Same as run 4 but encoder lr 2e-3 | 41.79% | — | Best at batch 11,999. Train loss ended around 18-30, still underfitting. Only change vs run 4 is lr_max (1e-3 → 2e-3); 11-point gain. T4, float32, compile off, 94 min, 0.48 s/batch. No prediction recorded before this run. |

## Training setup (all my runs)

- Kaggle T4 (one of two GPUs), float32 (`use_amp` false)
- Batch size 32 (official is 64; halved to avoid out-of-memory on a 15 GB T4)
- Cosine learning-rate schedule shortened to the number of batches; warmup is min(1000, 10% of batches)
- Official GRU architecture: 5 layers, 768 units, patch size 14, stride 4, day-specific input layers
- PER measured with greedy CTC decoding
- Rule: every model, including the transformer, uses the same batch size and number of batches

## Takeaways

- Reached 13.43% val PER with about 5% of the official run’s training samples (384k vs 7.68M), within 3.3 points of the official 10.1%.
- The transformer reached 52.61% val PER at matched batches (12,000 at batch size 32) vs 13.43% for the GRU. It underfit rather than overfit. Only one configuration was tried, so this shows it underperforms at this budget, not that transformers can’t work on this task.
- Doubling the transformer’s learning rate (1e-3 to 2e-3) cut val PER from 52.61% to 41.79%, so the first transformer result was largely a tuning problem, though it remains far behind the GRU’s 13.43%.

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
