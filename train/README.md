# Transformer trainer patch

`transformer_trainer.patch` modifies the official Brain-to-Text '25 baseline trainer so it can train `models/transformer.py` as well as the original GRU. This repo stores only the patch, not a copy of the official trainer.

## Credit

The trainer being patched is `model_training/rnn_trainer.py` from
[Neuroprosthetics-Lab/nejm-brain-to-text at commit 980a5b7](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/980a5b7e96392210edf976d76bdb5d15d2e0e62b),
written by Nick Card and Zachery Fogg.

That repository accompanies the paper "An Accurate and Rapidly Calibrating Speech Neuroprosthesis", *The New England Journal of Medicine* (2024), by Nicholas S. Card, Maitreyee Wairagkar, Carrina Iacobacci, Xianda Hou, Tyler Singer-Clark, Francis R. Willett, Erin M. Kunz, Chaofei Fan, Maryam Vahdati Nia, Darrel R. Deo, Aparna Srinivasan, Eun Young Choi, Matthew F. Glasser, Leigh R. Hochberg, Jaimie M. Henderson, Kiarash Shahlaie, Sergey D. Stavisky and David M. Brandman.

**License:** the official repository has no license file (checked at commit 980a5b7), so its code is not covered by any open-source license. That is why this repo keeps only a diff.

## What the patch changes

1. **Model choice:** if `model.type` is `transformer`, the trainer builds `TransformerDecoder` from `models/transformer.py`. Otherwise it builds the GRU exactly as before.
2. **Padding lengths:** a new `run_model` method passes the valid patch counts (`adjusted_lens`) to the transformer, so attention ignores padding. The GRU is called as before. Both `train()` and `validation()` use it.
3. **Compile switch:** `torch.compile` runs only if `compile` is true (the default).

Nothing else changes.

## Applying the patch

From the root of this repo, with the official repo cloned next to it:

```bash
git clone https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text.git ../nejm-brain-to-text
git -C ../nejm-brain-to-text checkout 980a5b7e96392210edf976d76bdb5d15d2e0e62b
patch -p1 -d ../nejm-brain-to-text < train/transformer_trainer.patch
```

The patched trainer imports `dataset`, `data_augmentations` and `rnn_model` from the official `model_training` folder, and `models.transformer` from this repo, so both folders must be on `sys.path`:

```python
import sys
sys.path.insert(0, "/path/to/nejm-brain-to-text/model_training")
sys.path.insert(0, "/path/to/speech-bci-transformer")
```

## Config keys the patch reads

Added to the official `rnn_args.yaml`. The GRU ignores the `transformer` block.

```yaml
compile: true          # optional, default true
model:
  type: transformer    # optional, default gru
  transformer:
    d_model: 384
    n_layers: 4
    n_heads: 6
    dim_feedforward: 1536
    dropout: 0.1
```

The transformer shares `n_input_features`, `input_network.input_layer_dropout`, `patch_size`, `patch_stride`, `dataset.sessions` and `dataset.n_classes` with the GRU.
