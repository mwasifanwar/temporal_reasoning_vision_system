import torch
import cv2
import numpy as np
from typing import List, Optional
import torchvision.transforms as transforms

class VideoProcessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224), frame_rate: int = 30):
        self.target_size = target_size
        self.frame_rate = frame_rate
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is not None:
            frame_indices = self._sample_frame_indices(total_frames, max_frames)
        else:
            frame_indices = list(range(total_frames))
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if i in frame_indices:
                processed_frame = self._preprocess_frame(frame)
                frames.append(processed_frame)
                frame_count += 1
        
        cap.release()
        
        if not frames:
            return torch.zeros(1, 16, 3, self.target_size[0], self.target_size[1])
        
        frames_tensor = torch.stack(frames).unsqueeze(0)
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
        
        tensor_frame = self.transform(frame_resized)
        return tensor_frame
    
    def extract_optical_flow(self, video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        
        flow_frames = []
        prev_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_magnitude = cv2.resize(flow_magnitude, self.target_size)
                flow_tensor = torch.tensor(flow_magnitude, dtype=torch.float32).unsqueeze(0)
                flow_frames.append(flow_tensor)
            
            prev_frame = gray
            frame_count += 1
        
        cap.release()
        
        if not flow_frames:
            return torch.zeros(1, 15, 1, self.target_size[0], self.target_size[1])
        
        flow_tensor = torch.stack(flow_frames).unsqueeze(0)
        return flow_tensor
    
    def compute_temporal_segments(self, video_path: str, num_segments: int = 8) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        segment_duration = total_frames / (fps * num_segments)
        segments = []
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            
            segment_frames = self.extract_frames_from_segment(cap, start_time, end_time)
            
            segment_info = {
                "segment_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "frame_count": len(segment_frames),
                "key_frames": self._extract_key_frames(segment_frames)
            }
            segments.append(segment_info)
        
        cap.release()
        return segments
    
    def extract_frames_from_segment(self, cap: cv2.VideoCapture, start_time: float, end_time: float) -> List[np.ndarray]:
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_pos in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        return frames
    
    def _extract_key_frames(self, frames: List[np.ndarray], num_key_frames: int = 3) -> List[int]:
        if len(frames) <= num_key_frames:
            return list(range(len(frames)))
        
        frame_differences = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            frame_differences.append(diff)
        
        key_frame_indices = [0]
        threshold = np.percentile(frame_differences, 80)
        
        for i, diff in enumerate(frame_differences):
            if diff > threshold:
                key_frame_indices.append(i + 1)
        
        if len(key_frame_indices) > num_key_frames:
            key_frame_indices = key_frame_indices[:num_key_frames]
        
        return key_frame_indices
