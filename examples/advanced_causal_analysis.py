import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import TemporalReasoningEngine
from training import TemporalReasoningTrainer, TemporalLoss
from data import VideoDataset
from torch.utils.data import DataLoader

def advanced_causal_analysis_demo():
    print("=== Advanced Causal Analysis and Temporal Reasoning Demo ===")
    print("Complex causal chain extraction and long-term event prediction")
    print("Created by mwasifanwar")
    
    engine = TemporalReasoningEngine(model_type="spatiotemporal")
    
    train_dataset = VideoDataset("data/train", "data/train_annotations.json")
    val_dataset = VideoDataset("data/val", "data/val_annotations.json")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.001)
    criterion = TemporalLoss()
    
    trainer = TemporalReasoningTrainer(
        model=engine,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    print("Starting temporal reasoning training...")
    trainer.train(num_epochs=5, save_path="best_temporal_model.pth")
    
    print("Training completed. Performing advanced causal analysis...")
    
    test_video = "test_video.mp4"
    advanced_results = engine.analyze_causal_chains(test_video)
    
    print("Extracted Causal Chains:")
    for i, chain in enumerate(advanced_results.get("causal_chains", [])[:2]):
        print(f"  Chain {i+1}: {len(chain)} events")
        for event in chain[:3]:
            print(f"    Frame {event['frame_index']} (root: {event['is_root']}, "
                  f"causes_next: {event['causes_next']})")
    
    print("\nLong-term Event Predictions:")
    for i, prediction in enumerate(advanced_results.get("event_predictions", [])[:2]):
        print(f"  Step {prediction['step']}: Uncertainty {prediction['uncertainty']:.3f}")
        for event in prediction['events'][:2]:
            print(f"    {event['event']} (confidence: {event['confidence']:.3f})")
    
    return engine, trainer

if __name__ == "__main__":
    engine, trainer = advanced_causal_analysis_demo()