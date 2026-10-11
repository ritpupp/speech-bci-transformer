import math

import torch
from torch import nn


def frames_to_patches(n_frames, patch_size=14, patch_stride=4):
    '''
    Number of patches that unfold produces from n_frames timesteps.
    Use this to turn per-trial frame counts into the `lengths` that forward expects.
    '''
    return (n_frames - patch_size) // patch_stride + 1


class SinusoidalPositionalEncoding(nn.Module):
    '''
    Fixed (not learned) sine/cosine position signal, added to each patch so attention knows the order.
    Computed for whatever sequence length comes in, so there is no maximum trial length.
    '''
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: [batches, num_patches, d_model]
        n_patches = x.size(1)
        position = torch.arange(n_patches, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(n_patches, self.d_model, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe


class TransformerDecoder(nn.Module):
    '''
    Transformer counterpart to the official GRUDecoder (nejm-brain-to-text/model_training/rnn_model.py).

    The day-specific input layers and patching are copied from GRUDecoder; only the GRU is replaced
    with a projection, positional encoding, and a pre-norm transformer encoder.
    '''
    def __init__(self,
                 neural_dim = 512,
                 n_days = 45,
                 n_classes = 41,
                 d_model = 384,
                 n_layers = 4,
                 n_heads = 6,
                 dim_feedforward = 1536,
                 dropout = 0.1,
                 input_dropout = 0.2,
                 patch_size = 14,
                 patch_stride = 4,
                 ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.d_model = d_model
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # --- Front end, copied from GRUDecoder ---
        self.day_layer_activation = nn.Softsign()

        # Identity init so each day's layer starts as "change nothing"
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim * self.patch_size

        # --- New part: projection, positional encoding, transformer ---
        self.input_projection = nn.Linear(self.input_size, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight)  # matches GRU's weight_ih init

        self.positional_encoding = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = n_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = "gelu",
            batch_first = True,
            norm_first = True,  # pre-norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            norm = nn.LayerNorm(d_model),  # pre-norm layers need a final norm
            enable_nested_tensor = False,
        )

        # Prediction head. Weight init to xavier, as in GRUDecoder
        self.output = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x, day_idx, lengths = None):
        '''
        x        (tensor)  - batch of trials, shape (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - day index for each trial in the batch
        lengths  (tensor)  - optional number of valid patches per trial; patches past this are padding
                             and are hidden from attention. Use frames_to_patches() to compute it.

        returns logits of shape (batch_size, num_patches, n_classes)
        '''

        # --- Day-specific layer, copied from GRUDecoder ---
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # --- Patching, copied from GRUDecoder ---
        x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
        x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]

        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]

        x_unfold = x_unfold.squeeze(2)           # [batches, feature_dim, num_patches, patch_size]
        x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

        x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)  # [batches, num_patches, patch_size * feature_dim]

        # --- Transformer ---
        x = self.input_projection(x)             # [batches, num_patches, d_model]
        x = self.positional_encoding(x)

        num_patches = x.size(1)
        padding_mask = None
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=x.device)
            assert (lengths <= num_patches).all(), \
                f"lengths {lengths.tolist()} exceed num_patches {num_patches}"
            assert (lengths >= 1).all(), \
                f"every trial needs at least one valid patch, got lengths {lengths.tolist()}"
            # True = padding, which attention ignores
            padding_mask = torch.arange(num_patches, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        return self.output(x)


def count_parameters(model):
    '''
    Print a per-part parameter breakdown for a TransformerDecoder and return the total.
    '''
    groups = {
        "day layers": ("day_weights", "day_biases"),
        "input projection": ("input_projection",),
        "transformer encoder": ("encoder",),
        "output layer": ("output",),
    }
    counts = {name: 0 for name in groups}
    for param_name, param in model.named_parameters():
        for group, prefixes in groups.items():
            if param_name.split(".")[0] in prefixes:
                counts[group] += param.numel()
                break

    total = sum(p.numel() for p in model.parameters())
    for group, n in counts.items():
        print(f"{group:<22}{n:>12,}")
    print(f"{'total':<22}{total:>12,}")
    return total
