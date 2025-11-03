import torch
from torch.utils.data import Dataset
import os
import cv2
import json
from typing import List, Dict, Optional

class VideoDataset(Dataset):
    def __init__(self, 
                 video_dir: str, 
                 annotation_file: str,
                 max_frames: int = 100,
                 transform=None):
        self.video_dir = video_dir
        self.annotation_file = annotation_file
        self.max_frames = max_frames
        self.transform = transform
        
        self.videos = self._load_annotations()
        self.preprocessor = VideoPreprocessor()
    
    def _load_annotations(self) -> List[Dict]:
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                annotations = json.load(f)
        else:
            annotations = self._generate_dummy_annotations()
        
        return annotations
    
    def _generate_dummy_annotations(self) -> List[Dict]:
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        annotations = []
        for video_file in video_files:
            annotation = {
                'video_path': os.path.join(self.video_dir, video_file),
                'actions': ['walking', 'running', 'jumping'][:2],
                'temporal_segments': [
                    {'start': 0, 'end': 30, 'action': 'walking'},
                    {'start': 31, 'end': 60, 'action': 'running'}
                ],
                'causal_relations': [
                    {'cause': 0, 'effect': 1, 'relation': 'enables'}
                ]
            }
            annotations.append(annotation)
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict:
        video_info = self.videos[idx]
        video_path = video_info['video_path']
        
        frames = self.preprocessor.extract_frames(video_path, self.max_frames)
        
        sample = {
            'frames': frames,
            'video_path': video_path,
            'actions': video_info.get('actions', []),
            'temporal_segments': video_info.get('temporal_segments', []),
            'causal_relations': video_info.get('causal_relations', [])
        }
        
        return sample