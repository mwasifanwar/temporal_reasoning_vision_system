import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2

class TemporalReasoningEngine:
    def __init__(self, model_type: str = "transformer"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.video_processor = VideoProcessor()
        
        if model_type == "transformer":
            self.temporal_model = TemporalTransformer()
        elif model_type == "spatiotemporal":
            self.temporal_model = SpatioTemporalModel()
        
        self.causal_reasoner = CausalReasoner()
        self.event_predictor = EventPredictor()
        
        self._move_to_device()
        
    def _move_to_device(self):
        self.temporal_model.to(self.device)
        self.causal_reasoner.to(self.device)
        self.event_predictor.to(self.device)
    
    def process_video(self, video_path: str, reasoning_tasks: List[str] = None) -> Dict:
        if reasoning_tasks is None:
            reasoning_tasks = ["action_recognition", "causal_analysis", "event_prediction"]
        
        frames = self.video_processor.extract_frames(video_path)
        temporal_features = self.temporal_model(frames)
        
        results = {}
        
        if "action_recognition" in reasoning_tasks:
            results["actions"] = self._recognize_actions(temporal_features, frames)
        
        if "causal_analysis" in reasoning_tasks:
            results["causal_relations"] = self.causal_reasoner.analyze_causality(temporal_features, frames)
        
        if "event_prediction" in reasoning_tasks:
            results["future_events"] = self.event_predictor.predict_events(temporal_features, frames)
        
        if "temporal_relationships" in reasoning_tasks:
            results["temporal_relations"] = self._extract_temporal_relationships(temporal_features)
        
        return results
    
    def _recognize_actions(self, temporal_features: torch.Tensor, frames: List) -> List[Dict]:
        batch_size, seq_len, feature_dim = temporal_features.shape
        
        action_logits = self.temporal_model.action_classifier(temporal_features)
        action_probs = F.softmax(action_logits, dim=-1)
        action_predictions = torch.argmax(action_probs, dim=-1)
        
        actions = []
        for i in range(seq_len):
            action_info = {
                "frame_index": i,
                "action_class": int(action_predictions[0, i].item()),
                "confidence": float(action_probs[0, i, action_predictions[0, i]].item()),
                "timestamp": i / 30.0
            }
            actions.append(action_info)
        
        return actions
    
    def _extract_temporal_relationships(self, temporal_features: torch.Tensor) -> List[Dict]:
        seq_len = temporal_features.shape[1]
        
        relationships = []
        for i in range(seq_len - 1):
            for j in range(i + 1, min(i + 5, seq_len)):
                relation = self._compute_temporal_relation(
                    temporal_features[:, i], 
                    temporal_features[:, j]
                )
                relationships.append({
                    "source_frame": i,
                    "target_frame": j,
                    "relation_type": relation,
                    "temporal_distance": j - i
                })
        
        return relationships
    
    def _compute_temporal_relation(self, feat1: torch.Tensor, feat2: torch.Tensor) -> str:
        similarity = F.cosine_similarity(feat1, feat2, dim=-1).item()
        
        if similarity > 0.8:
            return "simultaneous"
        elif similarity > 0.5:
            return "sequential"
        else:
            return "causal"
    
    def analyze_causal_chains(self, video_path: str) -> Dict:
        frames = self.video_processor.extract_frames(video_path)
        temporal_features = self.temporal_model(frames)
        
        causal_chains = self.causal_reasoner.extract_causal_chains(temporal_features)
        event_predictions = self.event_predictor.predict_long_term(temporal_features)
        
        return {
            "causal_chains": causal_chains,
            "event_predictions": event_predictions,
            "temporal_structure": self._analyze_temporal_structure(temporal_features)
        }
    
    def _analyze_temporal_structure(self, temporal_features: torch.Tensor) -> Dict:
        seq_len = temporal_features.shape[1]
        
        temporal_dependencies = []
        for i in range(seq_len):
            if i > 0:
                dependency = self._compute_temporal_dependency(
                    temporal_features[:, :i], 
                    temporal_features[:, i]
                )
                temporal_dependencies.append({
                    "frame": i,
                    "dependency_strength": dependency,
                    "influenced_by": list(range(i))
                })
        
        return {
            "temporal_dependencies": temporal_dependencies,
            "sequence_complexity": self._compute_sequence_complexity(temporal_features),
            "temporal_consistency": self._compute_temporal_consistency(temporal_features)
        }
    
    def _compute_temporal_dependency(self, past_features: torch.Tensor, current_feature: torch.Tensor) -> float:
        past_mean = past_features.mean(dim=1)
        correlation = F.cosine_similarity(past_mean, current_feature, dim=-1)
        return correlation.item()
    
    def _compute_sequence_complexity(self, temporal_features: torch.Tensor) -> float:
        differences = temporal_features[:, 1:] - temporal_features[:, :-1]
        complexity = torch.norm(differences, dim=-1).mean().item()
        return complexity
    
    def _compute_temporal_consistency(self, temporal_features: torch.Tensor) -> float:
        seq_len = temporal_features.shape[1]
        consistencies = []
        
        for i in range(1, seq_len):
            consistency = F.cosine_similarity(
                temporal_features[:, i-1], 
                temporal_features[:, i], 
                dim=-1
            ).item()
            consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 1.0