import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np

class TemporalReasoningTrainer:
    def __init__(self, 
                 model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            frames = batch['frames'].to(self.device)
            
            temporal_features = self.model.temporal_model(frames)
            
            action_predictions = self.model.temporal_model.action_classifier(temporal_features)
            
            causal_relations = self.model.causal_reasoner(temporal_features)
            
            event_predictions = self.model.event_predictor(temporal_features)
            
            loss = self.criterion(
                action_predictions,
                causal_relations,
                event_predictions,
                batch
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch['frames'].to(self.device)
                
                temporal_features = self.model.temporal_model(frames)
                
                action_predictions = self.model.temporal_model.action_classifier(temporal_features)
                
                causal_relations = self.model.causal_reasoner(temporal_features)
                
                event_predictions = self.model.event_predictor(temporal_features)
                
                loss = self.criterion(
                    action_predictions,
                    causal_relations,
                    event_predictions,
                    batch
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        return avg_loss
    
    def train(self, num_epochs: int, save_path: str = None):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if save_path and val_loss == self.best_val_loss:
                self.save_checkpoint(save_path, epoch)
    
    def save_checkpoint(self, path: str, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
