import pytest
import torch
from torch import nn

from models.transformer import TransformerDecoder, frames_to_patches

DAY_IDX = [0, 5]
N_REAL_FRAMES = 1000
N_PAD_FRAMES = 200
N_REAL_PATCHES = frames_to_patches(N_REAL_FRAMES)  # 247


def make_eval_model():
    '''
    Model in eval mode (no dropout) with random biases. At init every bias is zero, so appended zero
    frames would stay zero through the day layer; random biases make the padded patches nonzero.
    '''
    torch.manual_seed(0)
    model = TransformerDecoder()
    with torch.no_grad():
        for bias in model.day_biases:
            nn.init.normal_(bias)
        nn.init.normal_(model.input_projection.bias)
        nn.init.normal_(model.output.bias)
    model.eval()
    return model


def make_real_and_padded():
    torch.manual_seed(1)
    real = torch.randn(2, N_REAL_FRAMES, 512)
    padded = torch.cat([real, torch.zeros(2, N_PAD_FRAMES, 512)], dim=1)
    return real, padded


def test_output_shape():
    torch.manual_seed(0)
    model = TransformerDecoder()
    x = torch.randn(2, 1000, 512)
    logits = model(x, DAY_IDX)
    assert logits.shape == (2, 247, 41)


def test_padding_is_ignored():
    model = make_eval_model()
    real, padded = make_real_and_padded()
    with torch.no_grad():
        expected = model(real, DAY_IDX)
        out = model(padded, DAY_IDX, lengths=[N_REAL_PATCHES, N_REAL_PATCHES])
    assert out.shape[1] == frames_to_patches(N_REAL_FRAMES + N_PAD_FRAMES)
    assert torch.allclose(expected, out[:, :N_REAL_PATCHES], atol=1e-5)


def test_negative_control_without_lengths():
    # Same padded input but no mask: attention sees the padding, so the real outputs must change.
    # If this passed allclose, test_padding_is_ignored could not detect a broken mask.
    model = make_eval_model()
    real, padded = make_real_and_padded()
    with torch.no_grad():
        expected = model(real, DAY_IDX)
        out = model(padded, DAY_IDX)
    assert not torch.allclose(expected, out[:, :N_REAL_PATCHES], atol=1e-5)


def test_lengths_longer_than_num_patches_fails():
    model = make_eval_model()
    real, _ = make_real_and_padded()
    with pytest.raises(AssertionError):
        model(real, DAY_IDX, lengths=[N_REAL_PATCHES + 1, N_REAL_PATCHES])
