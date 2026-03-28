from typing import Tuple
import math

def get_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def get_width(bbox: Tuple[float, float, float, float]) -> float:
    x1, _, x2, _ = bbox
    return x2 - x1

def Euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    distance=math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return distance

def coordinate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return p1[0] - p2[0], p1[1] - p2[1]

def feet_position(bbox: Tuple[float, float, float, float]) -> Tuple[float, int]:
    x1, _, x2, y2 = bbox
    position=(x1 + x2) / 2, int(y2)
    return position