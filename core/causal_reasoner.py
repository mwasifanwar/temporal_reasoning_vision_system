import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

class CausalReasoner(nn.Module):
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_causal_relations: int = 5):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_causal_relations = num_causal_relations
        
        self.causal_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_causal_relations)
        )
        
        self.causal_strength_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.intervention_predictor = InterventionPredictor(feature_dim, hidden_dim)
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = temporal_features.shape
        
        causal_scores = []
        for i in range(seq_len - 1):
            for j in range(i + 1, seq_len):
                cause_effect_pair = torch.cat([
                    temporal_features[:, i],
                    temporal_features[:, j]
                ], dim=-1)
                
                encoded = self.causal_encoder(cause_effect_pair)
                relation_logits = self.relation_classifier(encoded)
                strength = self.causal_strength_estimator(encoded)
                
                causal_scores.append({
                    'cause_idx': i,
                    'effect_idx': j,
                    'relation_logits': relation_logits,
                    'strength': strength
                })
        
        return causal_scores
    
    def analyze_causality(self, temporal_features: torch.Tensor, frames: List) -> List[Dict]:
        causal_scores = self.forward(temporal_features)
        
        causal_relations = []
        for score in causal_scores:
            cause_idx = score['cause_idx']
            effect_idx = score['effect_idx']
            relation_probs = F.softmax(score['relation_logits'], dim=-1)
            relation_type = torch.argmax(relation_probs, dim=-1).item()
            strength = score['strength'].item()
            
            causal_relation = {
                'cause_frame': cause_idx,
                'effect_frame': effect_idx,
                'relation_type': self._get_relation_name(relation_type),
                'strength': strength,
                'temporal_gap': effect_idx - cause_idx,
                'confidence': float(relation_probs[0, relation_type].item())
            }
            causal_relations.append(causal_relation)
        
        return causal_relations
    
    def extract_causal_chains(self, temporal_features: torch.Tensor) -> List[List[Dict]]:
        batch_size, seq_len, feature_dim = temporal_features.shape
        
        causal_chains = []
        visited = set()
        
        for start_idx in range(seq_len):
            if start_idx not in visited:
                chain = self._find_causal_chain(temporal_features, start_idx, visited)
                if len(chain) > 1:
                    causal_chains.append(chain)
        
        return causal_chains
    
    def _find_causal_chain(self, temporal_features: torch.Tensor, start_idx: int, visited: set) -> List[Dict]:
        chain = []
        current_idx = start_idx
        
        while current_idx is not None and current_idx not in visited:
            visited.add(current_idx)
            
            next_cause = self._find_strongest_cause(temporal_features, current_idx, visited)
            
            chain_entry = {
                'frame_index': current_idx,
                'is_root': current_idx == start_idx,
                'causes_next': next_cause is not None
            }
            chain.append(chain_entry)
            
            current_idx = next_cause
        
        return chain
    
    def _find_strongest_cause(self, temporal_features: torch.Tensor, effect_idx: int, visited: set) -> int:
        if effect_idx == 0:
            return None
        
        batch_size, seq_len, feature_dim = temporal_features.shape
        
        best_strength = 0.0
        best_cause = None
        
        for cause_idx in range(effect_idx):
            if cause_idx in visited:
                continue
            
            cause_effect_pair = torch.cat([
                temporal_features[:, cause_idx],
                temporal_features[:, effect_idx]
            ], dim=-1)
            
            encoded = self.causal_encoder(cause_effect_pair)
            strength = self.causal_strength_estimator(encoded).item()
            
            if strength > best_strength:
                best_strength = strength
                best_cause = cause_idx
        
        return best_cause if best_strength > 0.5 else None
    
    def _get_relation_name(self, relation_type: int) -> str:
        relation_names = {
            0: "direct_causation",
            1: "indirect_causation", 
            2: "enables",
            3: "prevents",
            4: "correlation"
        }
        return relation_names.get(relation_type, "unknown")

class InterventionPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.intervention_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.effect_predictor = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()
        )
    
    def forward(self, cause_features: torch.Tensor, effect_features: torch.Tensor, 
                intervention: torch.Tensor) -> torch.Tensor:
        intervention_input = torch.cat([
            cause_features,
            effect_features, 
            intervention
        ], dim=-1)
        
        encoded = self.intervention_encoder(intervention_input)
        predicted_effect = self.effect_predictor(encoded)
        
        return predicted_effect