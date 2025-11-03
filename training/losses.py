import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalLoss(nn.Module):
    def __init__(self, 
                 action_weight: float = 1.0,
                 causal_weight: float = 0.5,
                 event_weight: float = 0.3,
                 consistency_weight: float = 0.2):
        super().__init__()
        
        self.action_weight = action_weight
        self.causal_weight = causal_weight
        self.event_weight = event_weight
        self.consistency_weight = consistency_weight
        
        self.action_loss = nn.CrossEntropyLoss()
        self.causal_loss = CausalConsistencyLoss()
        self.event_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                action_predictions: torch.Tensor,
                causal_relations: List,
                event_predictions: Dict,
                batch: Dict) -> torch.Tensor:
        
        action_targets = self._get_action_targets(batch, action_predictions.shape[1])
        action_loss = self.action_loss(
            action_predictions.view(-1, action_predictions.size(-1)),
            action_targets.view(-1)
        )
        
        causal_loss = self.causal_loss(causal_relations, batch.get('causal_relations', []))
        
        event_targets = self._get_event_targets(batch, event_predictions['event_predictions'].shape[1])
        event_loss = self.event_loss(
            event_predictions['event_predictions'].view(-1, event_predictions['event_predictions'].size(-1)),
            event_targets.view(-1)
        )
        
        consistency_loss = self._compute_temporal_consistency(action_predictions)
        
        total_loss = (self.action_weight * action_loss +
                     self.causal_weight * causal_loss +
                     self.event_weight * event_loss +
                     self.consistency_weight * consistency_loss)
        
        return total_loss
    
    def _get_action_targets(self, batch: Dict, seq_len: int) -> torch.Tensor:
        batch_size = batch['frames'].shape[0]
        return torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    def _get_event_targets(self, batch: Dict, pred_horizon: int) -> torch.Tensor:
        batch_size = batch['frames'].shape[0]
        return torch.zeros(batch_size, pred_horizon, dtype=torch.long)
    
    def _compute_temporal_consistency(self, action_predictions: torch.Tensor) -> torch.Tensor:
        seq_len = action_predictions.shape[1]
        
        if seq_len <= 1:
            return torch.tensor(0.0, device=action_predictions.device)
        
        predictions_diff = action_predictions[:, 1:] - action_predictions[:, :-1]
        consistency_loss = torch.mean(torch.abs(predictions_diff))
        
        return consistency_loss

class CausalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predicted_relations: List, target_relations: List) -> torch.Tensor:
        if not predicted_relations or not target_relations:
            return torch.tensor(0.0)
        
        loss = 0.0
        count = 0
        
        for pred_rel in predicted_relations:
            for target_rel in target_relations:
                if (pred_rel['cause_idx'] == target_rel['cause'] and 
                    pred_rel['effect_idx'] == target_rel['effect']):
                    loss += F.mse_loss(pred_rel['strength'], torch.tensor(1.0))
                    count += 1
        
        return loss / max(count, 1)