import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Dict


class KeypointsTracker:
    def __init__(self, model_path: str, conf: float = 0.1, kp_conf: float = 0.7) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        self.kp_conf = kp_conf
        
        self.original_size = (1920, 1080) 
        self.input_size = 640  #trained model input size

        #scaling factors to convert back to original size
        self.scale_x = self.original_size[0] / self.input_size
        self.scale_y = self.original_size[1] / self.input_size

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        # predict keypoints on resized frames
        resized_frames = [cv2.resize(f, (self.input_size, self.input_size)) for f in frames]
        return self.model.predict(resized_frames, conf=self.conf)

    def track(self, detection: Results) -> Dict:
        # standard supervision conversion
        kp_data = sv.KeyPoints.from_ultralytics(detection)
        
        if not kp_data or len(kp_data.xy) == 0:
            return {}

        xy = kp_data.xy[0] 
        confidence = kp_data.confidence[0]

        filtered_keypoints = {}
        for i, (coords, conf) in enumerate(zip(xy, confidence)):
            if conf > self.kp_conf:
                # Scale back to 1920x1080
                real_x = coords[0] * self.scale_x
                real_y = coords[1] * self.scale_y
                filtered_keypoints[i] = (real_x, real_y)

        return filtered_keypoints