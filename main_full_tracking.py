from utils.video import read_video, save_video, VideoProcessor, VideoWriter
from tracker.tracker import Tracker
from tracker.keypoints_tracker import KeypointsTracker
import cv2
import numpy as np
import os
from collections import deque

# ==============================================================================
# 1. EXACT FIELD GEOMETRY
# ==============================================================================
FIELD_CONNECTIONS = [
    (1, 9), (9, 10), (10, 11), (11, 12), (12, 4),
    (2, 6), (6, 7), (7, 3), (2, 3),
    (0, 1), (1, 2), (3, 4), (4, 5),

    (17, 18), (18, 19), (19, 20), (20, 28), (17, 25),
    (22, 23), (23, 27), (27, 26), (26, 22),
    (24, 25), (25, 28), (27, 29), (29, 26),

    (13, 14), (14, 15), (15, 16),
    (14, 31), (31, 15), (15, 30), (30, 14),

    (13, 24), (16, 29), (0, 13)
]

STANDARD_FIELD_COORDS = {
    0:(0,0),1:(0,13.84),2:(0,30.34),3:(0,37.66),4:(0,54.16),5:(0,68),
    6:(5.5,24.84),7:(5.5,43.16),8:(11,34),9:(16.5,13.84),
    10:(16.5,26.69),11:(16.5,41.31),12:(16.5,54.16),
    13:(52.5,0),14:(52.5,24.85),15:(52.5,43.15),16:(52.5,68),
    30:(43.35,34),31:(61.65,34),
    17:(88.5,13.84),18:(88.5,26.69),19:(88.5,41.31),20:(88.5,54.16),
    21:(94,34),22:(99.5,24.84),23:(99.5,43.16),
    24:(105,0),25:(105,13.84),26:(105,30.34),27:(105,37.66),
    28:(105,54.16),29:(105,68)
}

# ==============================================================================
# 2. HOMOGRAPHY ENGINE (NO AVERAGING)
# ==============================================================================
class ViewTransformer:
    def __init__(self):
        self.last_valid_H = None

    def update(self, detected_keypoints):
        src_pts, dst_pts = [], []
        for kid, coords in detected_keypoints.items():
            if kid in STANDARD_FIELD_COORDS:
                src_pts.append(coords)
                dst_pts.append(STANDARD_FIELD_COORDS[kid])
        
        if len(src_pts) < 4:
            return self.last_valid_H
            
        src_arr = np.array(src_pts, dtype=np.float32)
        dst_arr = np.array(dst_pts, dtype=np.float32)
        
        # COLLINEARITY GUARD
        x, y, w, h = cv2.boundingRect(src_arr)
        if w < 50 or h < 50: 
            return self.last_valid_H 
            
        H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        
        if H is not None and abs(np.linalg.det(H)) > 1e-6:
            self.last_valid_H = H
            
        return self.last_valid_H

# ==============================================================================
# 3. EMA PHYSICS SMOOTHER
# ==============================================================================
class PlayerEMAFilter:
    def __init__(self, alpha=0.3):
        self.positions = {}
        self.alpha = alpha

    def update(self, track_id, new_pos):
        if new_pos is None:
            return self.positions.get(track_id)
            
        if track_id not in self.positions:
            self.positions[track_id] = np.array(new_pos, dtype=np.float32)
            return new_pos
            
        smoothed = (self.alpha * np.array(new_pos, dtype=np.float32)) + ((1 - self.alpha) * self.positions[track_id])
        
        dist = np.linalg.norm(smoothed - self.positions[track_id])
        if dist > 3.0: 
            return self.positions[track_id]
            
        self.positions[track_id] = smoothed
        return smoothed

# ==============================================================================
# 4. HELPERS & VISUALIZATION
# ==============================================================================
def get_foot_position(bbox, is_ball=False):
    x1, y1, x2, y2 = bbox
    # If it's the ball, track the absolute center of the box
    if is_ball:
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        
    # If it's a player, anchor to their heels (bottom 5%)
    height = y2 - y1
    return np.array([(x1 + x2) / 2, y2 - (height * 0.05)], dtype=np.float32)

def transform_to_field_coords(pos, H):
    if H is None: return None
    pt = np.array([[pos]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

def draw_field_lines(frame, keypoints):
    if not keypoints: return frame
    
    # 1. Draw Yellow Lines (BGR: 0, 255, 255)
    for s, e in FIELD_CONNECTIONS:
        if s in keypoints and e in keypoints:
            cv2.line(frame, tuple(map(int, keypoints[s])), tuple(map(int, keypoints[e])), (0, 255, 255), 2)
            
    # 2. Draw Green Dots for Keypoints (BGR: 0, 255, 0)
    for kp_id, coords in keypoints.items():
        cv2.circle(frame, (int(coords[0]), int(coords[1])), 5, (0, 255, 0), -1)
        
    return frame

def draw_minimap(tracks, frame_index):
    m = np.ones((680, 1050, 3), dtype=np.uint8) * 40
    scale = 10
    
    cv2.rectangle(m, (0, 0), (1050, 680), (255, 255, 255), 2)
    cv2.line(m, (525, 0), (525, 680), (255, 255, 255), 2)
    cv2.circle(m, (525, 340), int(9.15 * scale), (255, 255, 255), 2)
    cv2.rectangle(m, (0, 138), (165, 542), (255, 255, 255), 2) 
    cv2.rectangle(m, (885, 138), (1050, 542), (255, 255, 255), 2) 
    cv2.rectangle(m, (0, 248), (55, 432), (255, 255, 255), 2)  
    cv2.rectangle(m, (995, 248), (1050, 432), (255, 255, 255), 2) 

    # Draw Players and Goalkeepers first
    colors = {"players": (0, 255, 0), "goalkeepers": (255, 0, 0)}
    for cat, col in colors.items():
        for tid, data in tracks[cat][frame_index].items():
            if data.get("field_pos") is not None:
                px, py = int(data["field_pos"][0]*scale), int(data["field_pos"][1]*scale)
                if -50 <= px <= 1100 and -50 <= py <= 730: 
                    cv2.circle(m, (px, py), 8, col, -1)
                    
    # Draw Ball LAST so it sits on top, with a high-visibility style
    for tid, data in tracks["ball"][frame_index].items():
        if data.get("field_pos") is not None:
            px, py = int(data["field_pos"][0]*scale), int(data["field_pos"][1]*scale)
            if -50 <= px <= 1100 and -50 <= py <= 730:
                cv2.circle(m, (px, py), 7, (0, 0, 0), -1)        # Bold Black Border
                cv2.circle(m, (px, py), 4, (255, 255, 255), -1)  # White Fill
                
    return m

def draw_combined_view(frame, tracks, kpts, f_idx, tracker):
    frame = draw_field_lines(frame, kpts)

    annotated = tracker.draw_annotations([frame], {
        k: [v[f_idx]] for k, v in tracks.items() if k != "keypoints"
    })[0]
    
    for cat in ["players", "goalkeepers"]:
        for tid, data in tracks[cat][f_idx].items():
            if data.get("field_pos") is not None:
                fx, fy = data["field_pos"]
                lx, ly = int((data["bbox"][0] + data["bbox"][2]) / 2), int(data["bbox"][1]) - 10
                label = f"({fx:.1f}, {fy:.1f})m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (lx - tw//2 - 2, ly - th - 2), (lx + tw//2 + 2, ly + 2), (0, 0, 0), -1)
                cv2.putText(annotated, label, (lx - tw//2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    m_map = draw_minimap(tracks, f_idx)
    m_small = cv2.resize(m_map, (350, 227)) 
    h, w = annotated.shape[:2]
    annotated[20:247, w-370:w-20] = m_small
    cv2.rectangle(annotated, (w-370, 20), (w-20, 247), (255, 255, 255), 2)
    return annotated

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    with VideoProcessor('input_videos/match.mp4') as video:
        p_tracker = Tracker('models/best.pt')
        k_tracker = KeypointsTracker('models/Keypoints_detection_best.pt', conf=0.3, kp_conf=0.5)

        def frame_gen():
            video.reset()
            for _, b in video.get_batch_generator(batch_size=1):
                yield b[0]

        p_tracks = p_tracker.get_object_tracks_chunked(frame_gen(), chunk_size=50)

        video.reset()
        all_kpts = []
        for _, batch in video.get_batch_generator(batch_size=10):
            dets = k_tracker.detect(batch)
            for d in dets: all_kpts.append(k_tracker.track(d))

        view_transformer = ViewTransformer()
        filters = {c: PlayerEMAFilter(alpha=0.25) for c in ["players", "goalkeepers", "referees"]}
        filters["ball"] = PlayerEMAFilter(alpha=0.6) 

        final_tracks = {c: [] for c in ["players", "goalkeepers", "referees", "ball"]}

        for f_idx in range(video.total_frames):
            H = view_transformer.update(all_kpts[f_idx])

            for cat in final_tracks:
                frame_data = {}
                for tid, d in p_tracks[cat][f_idx].items():
                    # FIX: Pass is_ball parameter to get absolute center for the ball
                    is_ball = (cat == "ball")
                    foot = get_foot_position(d["bbox"], is_ball=is_ball)
                    raw_pos = transform_to_field_coords(foot, H)
                    
                    safe_pos = filters[cat].update(tid, raw_pos)
                    
                    frame_data[tid] = {"bbox": d["bbox"], "field_pos": safe_pos}
                final_tracks[cat].append(frame_data)

        writer = VideoWriter('output_videos/full_tracking_final.mp4', video.width, video.height, video.fps)

        video.reset()
        for f_idx, (_, batch) in enumerate(video.get_batch_generator(batch_size=1)):
            annotated_frame = draw_combined_view(batch[0], final_tracks, all_kpts[f_idx], f_idx, p_tracker)
            writer.write(annotated_frame)

        writer.release()
        print("✅ SUCCESS: Added yellow lines, green keypoints, and distinct minimap ball!")

if __name__ == '__main__':
    main()