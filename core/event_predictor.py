import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

class EventPredictor(nn.Module):
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 prediction_horizon: int = 10,
                 num_event_classes: int = 20):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        self.num_event_classes = num_event_classes
        
        self.temporal_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.event_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_event_classes * prediction_horizon)
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, prediction_horizon),
            nn.Sigmoid()
        )
        
        self.trajectory_predictor = TrajectoryPredictor(feature_dim, hidden_dim, prediction_horizon)
    
    def forward(self, temporal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, feature_dim = temporal_features.shape
        
        encoded, (hidden, cell) = self.temporal_encoder(temporal_features)
        
        last_hidden = hidden[-1]
        
        event_predictions = self.event_predictor(last_hidden)
        event_predictions = event_predictions.view(batch_size, self.prediction_horizon, self.num_event_classes)
        
        uncertainties = self.uncertainty_estimator(last_hidden)
        
        trajectories = self.trajectory_predictor(temporal_features)
        
        return {
            'event_predictions': event_predictions,
            'uncertainties': uncertainties,
            'trajectories': trajectories,
            'encoded_features': encoded
        }
    
    def predict_events(self, temporal_features: torch.Tensor, frames: List) -> List[Dict]:
        predictions = self.forward(temporal_features)
        
        event_predictions = predictions['event_predictions']
        uncertainties = predictions['uncertainties']
        
        predicted_events = []
        for i in range(self.prediction_horizon):
            event_probs = F.softmax(event_predictions[0, i], dim=-1)
            event_class = torch.argmax(event_probs, dim=-1).item()
            confidence = event_probs[event_class].item()
            uncertainty = uncertainties[0, i].item()
            
            event_info = {
                'time_step': i + 1,
                'event_class': event_class,
                'event_name': self._get_event_name(event_class),
                'confidence': confidence,
                'uncertainty': uncertainty,
                'relative_time': (i + 1) / 30.0
            }
            predicted_events.append(event_info)
        
        return predicted_events
    
    def predict_long_term(self, temporal_features: torch.Tensor, num_steps: int = 5) -> List[Dict]:
        all_predictions = []
        current_features = temporal_features
        
        for step in range(num_steps):
            predictions = self.forward(current_features)
            
            step_prediction = {
                'step': step,
                'events': self._extract_step_events(predictions['event_predictions']),
                'trajectories': predictions['trajectories'],
                'uncertainty': predictions['uncertainties'].mean().item()
            }
            all_predictions.append(step_prediction)
            
            current_features = self._update_features(current_features, predictions)
        
        return all_predictions
    
    def _extract_step_events(self, event_predictions: torch.Tensor) -> List[str]:
        event_probs = F.softmax(event_predictions[0, 0], dim=-1)
        top_events = torch.topk(event_probs, 3)
        
        events = []
        for i in range(3):
            event_class = top_events.indices[i].item()
            confidence = top_events.values[i].item()
            events.append({
                'event': self._get_event_name(event_class),
                'confidence': confidence
            })
        
        return events
    
    def _update_features(self, current_features: torch.Tensor, predictions: Dict) -> torch.Tensor:
        batch_size, seq_len, feature_dim = current_features.shape
        
        predicted_event_embedding = self._event_to_embedding(predictions['event_predictions'][:, 0])
        
        updated_features = torch.cat([
            current_features[:, 1:],
            predicted_event_embedding.unsqueeze(1)
        ], dim=1)
        
        return updated_features
    
    def _event_to_embedding(self, event_predictions: torch.Tensor) -> torch.Tensor:
        event_probs = F.softmax(event_predictions, dim=-1)
        event_embedding = torch.matmul(event_probs, self.event_embedding_matrix())
        return event_embedding
    
    def event_embedding_matrix(self) -> torch.Tensor:
        return torch.randn(self.num_event_classes, self.feature_dim)
    
    def _get_event_name(self, event_class: int) -> str:
        event_names = [
            "object_appearance", "object_disappearance", "movement_start", 
            "movement_stop", "interaction_start", "interaction_end",
            "state_change", "composition_change", "spatial_relation_change",
            "temporal_pattern", "cyclic_event", "random_event",
            "causal_sequence", "parallel_events", "mutually_exclusive",
            "enabling_event", "preventing_event", "correlated_events",
            "independent_events", "complex_event"
        ]
        return event_names[event_class] if event_class < len(event_names) else "unknown_event"

class TrajectoryPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, prediction_horizon: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon * 2)
        )
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = temporal_features.shape
        
        encoded_trajectory = self.trajectory_encoder(temporal_features.mean(dim=1))
        predicted_trajectory = self.trajectory_decoder(encoded_trajectory)
        
        return predicted_trajectory.view(batch_size, self.prediction_horizon, 2)