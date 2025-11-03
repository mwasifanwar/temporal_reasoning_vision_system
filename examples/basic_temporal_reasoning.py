import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import TemporalReasoningEngine

def basic_temporal_reasoning_demo():
    print("=== Temporal Reasoning Vision System Demo ===")
    print("Advanced computer vision with temporal understanding and causal reasoning")
    print("Created by mwasifanwar")
    
    engine = TemporalReasoningEngine(model_type="transformer")
    
    print("1. Basic Video Processing and Action Recognition")
    video_path = "sample_video.mp4"
    
    results = engine.process_video(video_path, reasoning_tasks=["action_recognition"])
    
    print("Detected Actions:")
    for action in results.get("actions", [])[:5]:
        print(f"  Frame {action['frame_index']}: Action {action['action_class']} "
              f"(confidence: {action['confidence']:.3f})")
    
    print("\n2. Causal Relationship Analysis")
    causal_results = engine.process_video(video_path, reasoning_tasks=["causal_analysis"])
    
    print("Causal Relations:")
    for relation in causal_results.get("causal_relations", [])[:3]:
        print(f"  Frame {relation['cause_frame']} -> Frame {relation['effect_frame']}: "
              f"{relation['relation_type']} (strength: {relation['strength']:.3f})")
    
    print("\n3. Event Prediction")
    prediction_results = engine.process_video(video_path, reasoning_tasks=["event_prediction"])
    
    print("Predicted Future Events:")
    for event in prediction_results.get("future_events", [])[:3]:
        print(f"  Time +{event['time_step']}: {event['event_name']} "
              f"(confidence: {event['confidence']:.3f})")
    
    return engine

if __name__ == "__main__":
    engine = basic_temporal_reasoning_demo()