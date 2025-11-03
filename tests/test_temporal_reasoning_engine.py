import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import TemporalReasoningEngine

def test_engine_initialization():
    engine = TemporalReasoningEngine()
    
    assert engine.temporal_model is not None
    assert engine.causal_reasoner is not None
    assert engine.event_predictor is not None
    assert engine.video_processor is not None

def test_temporal_feature_processing():
    engine = TemporalReasoningEngine()
    
    dummy_frames = torch.randn(1, 10, 3, 224, 224)
    temporal_features = engine.temporal_model(dummy_frames)
    
    assert temporal_features is not None
    assert temporal_features.dim() == 3

def test_causal_analysis():
    engine = TemporalReasoningEngine()
    
    dummy_features = torch.randn(1, 10, 512)
    causal_relations = engine.causal_reasoner.analyze_causality(dummy_features, [])
    
    assert isinstance(causal_relations, list)