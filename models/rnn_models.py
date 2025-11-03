import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, cell) = self.lstm(x)
        
        output = self.output_projection(lstm_out)
        
        return output

class GRUTemporalModel(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.temporal_attention = TemporalAttention(hidden_size * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, hidden = self.gru(x)
        
        attended = self.temporal_attention(gru_out)
        
        return attended
