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
        self.input_size = 640  # trained model input size

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        # Predict keypoints on squashed frames to match Roboflow training data
        resized_frames = [cv2.resize(f, (self.input_size, self.input_size)) for f in frames]
        results = self.model.predict(resized_frames, conf=self.conf, verbose=False)
        
        # Inject the true original frame shapes into the results so track() can scale correctly
        for res, f in zip(results, frames):
            res.my_orig_shape = f.shape[:2] # (height, width)
            
        return results

    def track(self, detection: Results) -> Dict:
        kp_data = sv.KeyPoints.from_ultralytics(detection)
        
        if not kp_data or len(kp_data.xy) == 0:
            return {}

        xy = kp_data.xy[0] 
        confidence = kp_data.confidence[0]

        # Dynamically scale back based on this specific frame's original size
        orig_h, orig_w = getattr(detection, 'my_orig_shape', (1080, 1920))
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        filtered_keypoints = {}
        for i, (coords, conf) in enumerate(zip(xy, confidence)):
            if conf > self.kp_conf:
                # Scale back to the exact original frame dimensions
                real_x = coords[0] * scale_x
                real_y = coords[1] * scale_y
                filtered_keypoints[i] = (real_x, real_y)

        return filtered_keypoints
