import torch
import torch.nn as nn
import math
import numpy as np



class SinusoidalTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 366, tau: float = 1000.0):
        super().__init__()

        pe = torch.zeros(1, max_len, d_model)

        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:  #even: sin
                    pe[0, pos, i] = math.sin(pos / (tau ** (2 * (i // 2) / d_model)))
                else:  #odd: cos
                    pe[0, pos, i] = math.cos(pos / (tau ** (2 * (i // 2) / d_model)))

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, day_indices: torch.Tensor) -> torch.Tensor:
        encodings = self.pe[:, day_indices, :]  #(1, T, Dmodel)
        return self.dropout(encodings)





#pixel Set Encoder

class PixelSetEncoder(nn.Module):
    def __init__(self, mlp1_dims: list, mlp2_dims: list):
        super().__init__()

        #mpl1
        mlp1_layers = []
        for i in range(len(mlp1_dims) - 1):
            mlp1_layers.append(nn.Linear(mlp1_dims[i], mlp1_dims[i + 1]))
            if i < len(mlp1_dims) - 2:
                mlp1_layers.append(nn.BatchNorm1d(mlp1_dims[i + 1]))
                mlp1_layers.append(nn.ReLU())
        self.mlp1 = nn.Sequential(*mlp1_layers)

        #mlp2
        mlp2_layers = []
        for i in range(len(mlp2_dims) - 1):
            mlp2_layers.append(nn.Linear(mlp2_dims[i], mlp2_dims[i + 1]))
            if i < len(mlp2_dims) - 2:  # No BatchNorm/ReLU on final layer
                mlp2_layers.append(nn.BatchNorm1d(mlp2_dims[i + 1]))
                mlp2_layers.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*mlp2_layers)

    def forward(self, x_s_t: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, C = x_s_t.shape

        #mpl1
        x_flat = x_s_t.reshape(B * S, C)
        mlp1_output = self.mlp1(x_flat)
        mlp1_output = mlp1_output.view(B, S, -1)  #(B, S, mlp1_out_dim)

        #mean and std along pixel dimension
        if mask is None:
            #train
            mean_pooled = torch.mean(mlp1_output, dim=1)
            std_pooled = torch.std(mlp1_output, dim=1)
        else:
            #eval zero out features of padded
            mask_expanded = mask.unsqueeze(-1)
            masked_features = mlp1_output * mask_expanded

            #n of real pixels
            num_real_pixels = mask.sum(dim=1, keepdim=True).float()

            #masked mean
            mean_pooled = masked_features.sum(dim=1) / num_real_pixels

            #std
            mean_expanded = mean_pooled.unsqueeze(1)
            squared_diff = (masked_features - mean_expanded) ** 2
            masked_squared_diff = squared_diff * mask_expanded
            variance = masked_squared_diff.sum(dim=1) / num_real_pixels
            std_pooled = torch.sqrt(variance)

        pooled_features = torch.cat((mean_pooled, std_pooled), dim=1)

        e_t = self.mlp2(pooled_features)

        return e_t




#transformer encoder

class TransformerTemporalEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int,
                 day_of_year_sequence: np.ndarray, dropout_rate: float = 0.1,
                 tau_pe: float = 1000.0, max_len_pe: int = 366):
        super().__init__()

        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.positional_encoder = SinusoidalTemporalPositionalEncoding(
            d_model, dropout_rate, max_len_pe, tau_pe
        )

        self.register_buffer('day_of_year_sequence_tensor',
                           torch.tensor(day_of_year_sequence, dtype=torch.long))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, e_sequence: torch.Tensor) -> torch.Tensor:
        B, T, D = e_sequence.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  #(B, 1, d_model)
        sequence_with_cls = torch.cat([cls_tokens, e_sequence], dim=1)  #(B, T+1, d_model)

        cls_day_sequence = torch.cat([
            torch.tensor([0], device=self.day_of_year_sequence_tensor.device),
            self.day_of_year_sequence_tensor
        ])

        pos_encodings = self.positional_encoder(cls_day_sequence)
        sequence_with_pos = sequence_with_cls + pos_encodings

        transformer_out = self.transformer(sequence_with_pos)

        cls_representation = transformer_out[:, 0, :]  # (B, d_model)

        return cls_representation



class ClassificationHead(nn.Module):
    def __init__(self, mlp_dims: list, dropout_rate: float = 0.1):
        super().__init__()

        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
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

        self.pixel_set_encoder = PixelSetEncoder(**pse_config)
        self.temporal_transformer_encoder = TransformerTemporalEncoder(
            day_of_year_sequence=day_of_year_sequence, **transformer_config
        )
        self.classification_head = ClassificationHead(**classifier_config)

    def forward(self, x_pixel_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, T, C = x_pixel_data.shape

        x_pixel_data = x_pixel_data.float()


        B, S, T, C = x_pixel_data.shape

        #for better paralelization
        # (B, S, T, C) -> (B, T, S, C) -> (B * T, S, C)
        x_reshaped = x_pixel_data.permute(0, 2, 1, 3).reshape(B * T, S, C)

        mask_reshaped = None
        if mask is not None:
            mask_reshaped = mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, S)

        e_reshaped = self.pixel_set_encoder(x_reshaped, mask_reshaped)  #(B * T, d_e)

        #original format
        e_sequence = e_reshaped.view(B, T, -1)

        cls_token_features = self.temporal_transformer_encoder(e_sequence)

        logits = self.classification_head(cls_token_features)  # (B, num_classes)

        return logits