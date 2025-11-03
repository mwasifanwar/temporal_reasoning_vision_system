import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 max_seq_len: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 10)
        )
        
        self.temporal_attention = TemporalAttention(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = x.shape
        
        x_proj = self.input_projection(x)
        
        x_proj = x_proj.transpose(0, 1)
        x_proj = self.positional_encoding(x_proj)
        
        encoded = self.transformer_encoder(x_proj)
        encoded = encoded.transpose(0, 1)
        
        attended = self.temporal_attention(encoded)
        
        return attended

class SpatioTemporalModel(nn.Module):
    def __init__(self, 
                 spatial_dim: int = 2048,
                 temporal_dim: int = 512,
                 num_classes: int = 10):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, temporal_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.temporal_lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(temporal_dim * 2, temporal_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.attention_mechanism = SpatioTemporalAttention(temporal_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(temporal_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, spatial_dim = x.shape
        
        spatial_encoded = self.spatial_encoder(x)
        
        lstm_out, (hidden, cell) = self.temporal_lstm(spatial_encoded)
        
        lstm_out = lstm_out.transpose(1, 2)
        conv_out = self.temporal_conv(lstm_out)
        conv_out = conv_out.transpose(1, 2)
        
        attended = self.attention_mechanism(conv_out, spatial_encoded)
        
        final_representation = torch.cat([attended, hidden[-1]], dim=-1)
        
        return final_representation.unsqueeze(1).repeat(1, seq_len, 1)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended = torch.bmm(attention_weights, V)
        
        return attended

class SpatioTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.spatial_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, temporal_features: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = temporal_features.shape
        
        temporal_features = temporal_features.transpose(0, 1)
        spatial_features = spatial_features.transpose(0, 1)
        
        temporal_attended, _ = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        spatiotemporal_attended, _ = self.spatial_attention(
            temporal_attended, spatial_features, spatial_features
        )
        
        spatiotemporal_attended = spatiotemporal_attended.transpose(0, 1)
        
        output = self.layer_norm(spatiotemporal_attended.mean(dim=1))
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
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