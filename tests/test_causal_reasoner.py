import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CausalReasoner

def test_causal_reasoner_initialization():
    reasoner = CausalReasoner()
    
    assert reasoner.causal_encoder is not None
    assert reasoner.relation_classifier is not None
    assert reasoner.causal_strength_estimator is not None

def test_causal_relation_computation():
    reasoner = CausalReasoner()
    
    dummy_features = torch.randn(1, 5, 512)
    causal_scores = reasoner(dummy_features)
    
    assert isinstance(causal_scores, list)
    assert len(causal_scores) > 0

def test_causal_chain_extraction():
    reasoner = CausalReasoner()
    
    dummy_features = torch.randn(1, 8, 512)
    causal_chains = reasoner.extract_causal_chains(dummy_features)
    
    assert isinstance(causal_chains, list)