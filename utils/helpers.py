import logging
import json
import torch
import numpy as np
from datetime import datetime

def setup_logging(name: str = "temporal_reasoning"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def save_results(results: dict, filename: str = "temporal_reasoning_results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))

def calculate_temporal_metrics(predictions: dict, targets: dict) -> dict:
    metrics = {}
    
    if 'actions' in predictions and 'actions' in targets:
        action_accuracy = calculate_action_accuracy(predictions['actions'], targets['actions'])
        metrics['action_accuracy'] = action_accuracy
    
    if 'causal_relations' in predictions and 'causal_relations' in targets:
        causal_precision = calculate_causal_precision(predictions['causal_relations'], targets['causal_relations'])
        metrics['causal_precision'] = causal_precision
    
    if 'future_events' in predictions:
        event_confidence = calculate_event_confidence(predictions['future_events'])
        metrics['event_confidence'] = event_confidence
    
    return metrics

def calculate_action_accuracy(predicted_actions: list, target_actions: list) -> float:
    if not predicted_actions or not target_actions:
        return 0.0
    
    correct = 0
    total = min(len(predicted_actions), len(target_actions))
    
    for i in range(total):
        if predicted_actions[i]['action_class'] == target_actions[i]:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def calculate_causal_precision(predicted_relations: list, target_relations: list) -> float:
    if not predicted_relations:
        return 0.0
    
    true_positives = 0
    for pred_rel in predicted_relations:
        for target_rel in target_relations:
            if (pred_rel['cause_frame'] == target_rel['cause'] and 
                pred_rel['effect_frame'] == target_rel['effect']):
                true_positives += 1
                break
    
    return true_positives / len(predicted_relations) if predicted_relations else 0.0

def calculate_event_confidence(predicted_events: list) -> float:
    if not predicted_events:
        return 0.0
    
    confidences = [event['confidence'] for event in predicted_events]
    return np.mean(confidences)