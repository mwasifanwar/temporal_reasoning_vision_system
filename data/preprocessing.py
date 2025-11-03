import torch
import cv2
import numpy as np
from typing import List

class VideoPreprocessor:
    def __init__(self, target_size: tuple = (224, 224)):
        self.target_size = target_size
    
    def extract_frames(self, video_path: str, max_frames: int = 100) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = self._sample_frame_indices(total_frames, max_frames)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if i in frame_indices:
                processed_frame = self._preprocess_frame(frame)
                frames.append(processed_frame)
        
        cap.release()
        
        if not frames:
            return torch.zeros(max_frames, 3, self.target_size[0], self.target_size[1])
        
        frames_tensor = torch.stack(frames)
        return frames_tensor
    
    def _sample_frame_indices(self, total_frames: int, max_frames: int) -> List[int]:
        if total_frames <= max_frames:
            return list(range(total_frames))
        
        step = total_frames / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        return indices
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.target_size)
        
        tensor_frame = torch.from_numpy(frame_resized).float() / 255.0
        tensor_frame = tensor_frame.permute(2, 0, 1)
        
        return tensor_frame