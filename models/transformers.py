import torch
import torch.nn as nn
import math

class VideoTransformer(nn.Module):
    def __init__(self, 
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048):
        super().__init__()
        
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.temporal_encoding = TemporalEncoding(d_model)
        self.spatial_projection = nn.Linear(2048, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, spatial_dim = x.shape
        
        x_proj = self.spatial_projection(x)
        x_proj = x_proj.transpose(0, 1)
        
        x_proj = self.temporal_encoding(x_proj)
        
        encoded = self.transformer(x_proj)
        encoded = encoded.transpose(0, 1)
        
        return encoded

class TemporalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]