
"""
Parcel Time Series Classification Model for Agricultural Land Use Classification

This module implements a deep learning architecture for classifying agricultural parcels based on
multispectral satellite imagery time series data. The model is specifically designed for the
Timematch dataset format which contains parcels with variable numbers of pixels and fixed temporal
sequences.

ARCHITECTURE OVERVIEW:
The model follows a three-stage pipeline:
1. Pixel Set Encoder (PSE): Processes variable-sized pixel sets at each time step
2. Transformer Temporal Encoder: Models temporal dependencies across the time series
3. Classification Head: Produces final crop type predictions

DATA FORMAT EXPECTATIONS:
Input tensor shape: (B, S, T, C) where:
- B: batch size
- S: number of pixels per parcel (variable, but fixed within batch)
- T: number of time steps (typically 52 for weekly acquisitions)
- C: number of spectral channels (typically 10 for multispectral data)

CLASSES AND THEIR ROLES:

SinusoidalTemporalPositionalEncoding:
   Implements sinusoidal positional encoding for temporal sequences using day-of-year positions.
   Uses configurable temporal scaling parameter tau (default 1000.0) and supports sequences up to
   366 days. Applies dropout regularization and pre-computes encodings for efficiency.

   Key parameters:
   - d_model: embedding dimension
   - tau: temporal frequency scaling factor
   - max_len: maximum sequence length (366 for full year)

PixelSetEncoder:
   Processes variable-sized sets of pixels at each time step using a permutation-invariant approach.
   Uses two MLPs: first processes individual pixels, then applies statistical pooling (mean + std)
   across pixels, followed by a second MLP for final embedding.

   Architecture: MLP1 → Statistical Pooling → MLP2
   - MLP1: processes each pixel independently
   - Pooling: computes mean and standard deviation across pixels
   - MLP2: maps pooled features to final embedding dimension

   Both MLPs use BatchNorm + ReLU between layers (except final layer).

TransformerTemporalEncoder:
   Models temporal dependencies using a standard Transformer encoder with CLS token approach.
   Prepends a learnable CLS token to the temporal sequence, applies day-of-year based positional
   encoding, and processes through multi-layer transformer. The final CLS token representation
   captures the entire temporal sequence.

   Key features:
   - Learnable CLS token for sequence-level representation
   - Day-of-year positional encoding (CLS token gets position 0)
   - Standard PyTorch TransformerEncoder with configurable layers/heads
   - Batch-first processing for efficiency

ClassificationHead:
   Simple MLP for final classification with configurable depth and dropout regularization.
   Applies ReLU + Dropout between layers (except final layer which outputs raw logits).

ParcelTimeSeriesClassifier:
   Main model class that orchestrates the complete pipeline. Processes input through PSE at each
   time step, stacks embeddings into temporal sequence, applies Transformer encoder, and generates
   final predictions.

   Forward pass:
   1. For each time step t: apply PSE to pixel data at time t
   2. Stack time step embeddings into sequence
   3. Process sequence through Transformer (returns CLS token features)
   4. Apply classification head to get final logits

CONFIGURATION DICTIONARIES:
- pse_config: {"mlp1_dims": [C, ...], "mlp2_dims": [2*mlp1_out, d_e]}
- transformer_config: {"d_model": d_e, "n_heads": int, "num_layers": int, "dropout_rate": float}
- classifier_config: {"mlp_dims": [d_model, ..., num_classes], "dropout_rate": float}

TEMPORAL ENCODING:
The model requires a day_of_year_sequence array (typically length 52) containing the day-of-year
values for each acquisition date. This is used for temporal positional encoding and should be
derived from the dates.json file in the Timematch dataset format.

USAGE EXAMPLE:
model = ParcelTimeSeriesClassifier(
   pse_config={"mlp1_dims": [10, 64, 32], "mlp2_dims": [64, 128]},
   transformer_config={"d_model": 128, "n_heads": 8, "num_layers": 4, "dropout_rate": 0.1},
   classifier_config={"mlp_dims": [128, 64, num_classes], "dropout_rate": 0.1},
   day_of_year_sequence=day_of_year_array
)
logits = model(pixel_data)  # pixel_data shape: (B, S, T, C)
"""




import torch
import torch.nn as nn
import math
import numpy as np



class SinusoidalTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 366, tau: float = 1000.0):
        super().__init__()

        # Pre-compute positional encoding matrix
        pe = torch.zeros(1, max_len, d_model)

        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:  # Even indices: sin
                    pe[0, pos, i] = math.sin(pos / (tau ** (2 * (i // 2) / d_model)))
                else:  # Odd indices: cos
                    pe[0, pos, i] = math.cos(pos / (tau ** (2 * (i // 2) / d_model)))

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, day_indices: torch.Tensor) -> torch.Tensor:
        # Select encodings for the given day indices
        encodings = self.pe[:, day_indices, :]  # Shape: (1, Time_steps, d_model)
        return self.dropout(encodings)





# Pixel Set Encoder

class PixelSetEncoder(nn.Module):
    def __init__(self, mlp1_dims: list, mlp2_dims: list):
        super().__init__()

        # Build MLP1
        mlp1_layers = []
        for i in range(len(mlp1_dims) - 1):
            mlp1_layers.append(nn.Linear(mlp1_dims[i], mlp1_dims[i + 1]))
            if i < len(mlp1_dims) - 2:  # No BatchNorm/ReLU on final layer
                mlp1_layers.append(nn.BatchNorm1d(mlp1_dims[i + 1]))
                mlp1_layers.append(nn.ReLU())
        self.mlp1 = nn.Sequential(*mlp1_layers)

        # Build MLP2
        mlp2_layers = []
        for i in range(len(mlp2_dims) - 1):
            mlp2_layers.append(nn.Linear(mlp2_dims[i], mlp2_dims[i + 1]))
            if i < len(mlp2_dims) - 2:  # No BatchNorm/ReLU on final layer
                mlp2_layers.append(nn.BatchNorm1d(mlp2_dims[i + 1]))
                mlp2_layers.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*mlp2_layers)

    def forward(self, x_s_t: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, C = x_s_t.shape

        # Step 1: Apply MLP1 to each pixel
        x_flat = x_s_t.view(B * S, C)
        mlp1_output = self.mlp1(x_flat)  # (B*S, mlp1_out_dim)
        mlp1_output = mlp1_output.view(B, S, -1)  # (B, S, mlp1_out_dim)

        # Step 2: Pooling - compute mean and std along pixel dimension
        if mask is None:
            # Training case: standard pooling
            mean_pooled = torch.mean(mlp1_output, dim=1)  # (B, mlp1_out_dim)
            std_pooled = torch.std(mlp1_output, dim=1)  # (B, mlp1_out_dim)
        else:
            # Evaluation case: masked pooling
            # Zero out features of padded pixels
            mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)
            masked_features = mlp1_output * mask_expanded  # (B, S, mlp1_out_dim)

            # Calculate number of real pixels per batch item
            num_real_pixels = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
            num_real_pixels = torch.clamp(num_real_pixels, min=1.0)  # Avoid division by zero

            # Masked mean
            mean_pooled = masked_features.sum(dim=1) / num_real_pixels  # (B, mlp1_out_dim)

            # Masked standard deviation
            mean_expanded = mean_pooled.unsqueeze(1)  # (B, 1, mlp1_out_dim)
            squared_diff = (masked_features - mean_expanded) ** 2
            masked_squared_diff = squared_diff * mask_expanded
            variance = masked_squared_diff.sum(dim=1) / num_real_pixels  # (B, mlp1_out_dim)
            std_pooled = torch.sqrt(variance + 1e-8)  # Add small epsilon for numerical stability

        pooled_features = torch.cat((mean_pooled, std_pooled), dim=1)  # (B, 2*mlp1_out_dim)

        # Step 3: Apply MLP2
        e_t = self.mlp2(pooled_features)  # (B, d_e)

        return e_t




# Transformer Temporal Encoder with CLS Token

class TransformerTemporalEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int,
                 day_of_year_sequence: np.ndarray, dropout_rate: float = 0.1,
                 tau_pe: float = 1000.0, max_len_pe: int = 366):
        super().__init__()

        self.d_model = d_model

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.positional_encoder = SinusoidalTemporalPositionalEncoding(
            d_model, dropout_rate, max_len_pe, tau_pe
        )

        # Register day of year sequence as buffer (for temporal positions)
        self.register_buffer('day_of_year_sequence_tensor',
                           torch.tensor(day_of_year_sequence, dtype=torch.long))

        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, e_sequence: torch.Tensor) -> torch.Tensor:
        B, T, D = e_sequence.shape

        # Step 1: Add CLS token to beginning of sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        sequence_with_cls = torch.cat([cls_tokens, e_sequence], dim=1)  # (B, T+1, d_model)

        # Step 2: Create positional encoding indices
        # CLS token gets position 0, temporal steps get day-based positions
        cls_day_sequence = torch.cat([
            torch.tensor([0], device=self.day_of_year_sequence_tensor.device),
            self.day_of_year_sequence_tensor
        ])

        # Step 3: Add positional encoding
        pos_encodings = self.positional_encoder(cls_day_sequence)  # (1, T+1, d_model)
        sequence_with_pos = sequence_with_cls + pos_encodings

        # Step 4: Apply transformer
        transformer_out = self.transformer(sequence_with_pos)  # (B, T+1, d_model)

        # Step 5: Extract CLS token representation
        cls_representation = transformer_out[:, 0, :]  # (B, d_model)

        return cls_representation



# Final Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, mlp_dims: list, dropout_rate: float = 0.1):
        super().__init__()

        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # No ReLU/Dropout on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Main Model

class ParcelTimeSeriesClassifier(nn.Module):
    def __init__(self, pse_config: dict, transformer_config: dict, classifier_config: dict,
                 day_of_year_sequence: np.ndarray):
        super().__init__()

        # Initialize the three main components
        self.pixel_set_encoder = PixelSetEncoder(**pse_config)
        self.temporal_transformer_encoder = TransformerTemporalEncoder(
            day_of_year_sequence=day_of_year_sequence, **transformer_config
        )
        self.classification_head = ClassificationHead(**classifier_config)

    def forward(self, x_pixel_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, T, C = x_pixel_data.shape

        # Convert to float32 to match model parameters
        x_pixel_data = x_pixel_data.float()

        # A more efficient implementation for model.py in ParcelTimeSeriesClassifier.forward()

        B, S, T, C = x_pixel_data.shape

        # 1. Reshape the input to combine batch and time dimensions
        # (B, S, T, C) -> (B, T, S, C) -> (B * T, S, C)
        x_reshaped = x_pixel_data.permute(0, 2, 1, 3).reshape(B * T, S, C)

        # 2. Expand the mask to match the new reshaped input
        mask_reshaped = None
        if mask is not None:
            # The mask is the same for all time steps of a parcel.
            # We expand it from (B, S) to (B * T, S).
            mask_reshaped = mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, S)

        # 3. Apply the PixelSetEncoder only ONCE on the large batch
        e_reshaped = self.pixel_set_encoder(x_reshaped, mask_reshaped)  # Shape: (B * T, d_e)

        # 4. Reshape the output back to the desired sequence format
        # (B * T, d_e) -> (B, T, d_e)
        e_sequence = e_reshaped.view(B, T, -1)

        # Process sequence through Transformer with CLS token
        cls_token_features = self.temporal_transformer_encoder(e_sequence)  # (B, transformer_d_model)

        # Generate classification logits
        logits = self.classification_head(cls_token_features)  # (B, num_classes)

        return logits