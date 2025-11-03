import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, 
                 feature_dim: int,
                 num_heads: int = 8,
                 scales: List[int] = [1, 2, 4, 8]):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.scales = scales
        
        self.scale_attentions = nn.ModuleList([
            TemporalAttentionHead(feature_dim, num_heads, scale)
            for scale in scales
        ])
        
        self.fusion = nn.Linear(feature_dim * len(scales), feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = x.shape
        
        scale_outputs = []
        for attention in self.scale_attentions:
            scale_out = attention(x)
            scale_outputs.append(scale_out)
        
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.fusion(fused)
        
        return output

class TemporalAttentionHead(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, scale: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.scale = scale
        self.head_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.scale_factor = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.scale > 1:
            K = F.avg_pool1d(K.transpose(1, 2).contiguous().view(batch_size * seq_len, -1), 
                           self.scale, self.scale).view(batch_size, seq_len // self.scale, -1)
            K = K.view(batch_size, seq_len // self.scale, self.num_heads, self.head_dim).transpose(1, 2)
            V = F.avg_pool1d(V.transpose(1, 2).contiguous().view(batch_size * seq_len, -1), 
                           self.scale, self.scale).view(batch_size, seq_len // self.scale, -1)
            V = V.view(batch_size, seq_len // self.scale, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return attended