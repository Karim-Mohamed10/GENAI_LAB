from utils.video import read_video, save_video, VideoProcessor, VideoWriter
from tracker.tracker import Tracker
from tracker.keypoints_tracker import KeypointsTracker
import cv2
import numpy as np
import os
from collections import deque

# ==============================================================================
# 1. FIELD CONNECTIONS & EXACT COORDINATES
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
# 2. ENGINES: HOMOGRAPHY & PHYSICS FILTER
# ==============================================================================
class StableHomography:
    def __init__(self, buffer_size=7):
        self.buffer = deque(maxlen=buffer_size)

    def validate(self, H):
        if H is None: return False
        det = np.linalg.det(H)
        # Reject degenerate or inverted projections
        if abs(det) < 1e-6 or det < 0: return False
        return True

    def update(self, detected_keypoints):
        src_pts, dst_pts = [], []
        # Use ALL available points to give RANSAC the best chance
        for kid, (x, y) in detected_keypoints.items():
            if kid in STANDARD_FIELD_COORDS:
                src_pts.append([x, y])
                dst_pts.append(STANDARD_FIELD_COORDS[kid])
        
        if len(src_pts) >= 4:
            H, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, 5.0)
            if self.validate(H):
                self.buffer.append(H)
        
        if len(self.buffer) == 0:
            return None
            
        # Return smoothed matrix
        return np.mean(self.buffer, axis=0)

class PlayerPositionFilter:
    def __init__(self, max_movement_per_frame=2.0):
        self.last_positions = {}
        self.max_movement = max_movement_per_frame # Meters per frame limit

    def update(self, track_id, new_pos):
        if new_pos is None:
            return self.last_positions.get(track_id)
            
        if track_id not in self.last_positions:
            self.last_positions[track_id] = new_pos
            return new_pos
            
        # Check distance between last frame and current calculation
        dist = np.linalg.norm(np.array(new_pos) - np.array(self.last_positions[track_id]))
        
        if dist > self.max_movement:
            # Physics violation! Reject the new calculation and keep the old one.
            return self.last_positions[track_id]
            
        self.last_positions[track_id] = new_pos
        return new_pos

# ==============================================================================
# 3. HELPERS & VISUALIZATION
# ==============================================================================
def get_player_foot_position(bbox):
    return np.array([(bbox[0]+bbox[2])/2, bbox[3]], dtype=np.float32)

def transform_to_field_coords(pos, H):
    if H is None: return None
    pt = np.array([[pos]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

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

    colors = {"players": (0, 255, 0), "goalkeepers": (255, 0, 0), "ball": (255, 255, 255)}
    for cat, col in colors.items():
        for tid, data in tracks[cat][frame_index].items():
            if data.get("field_pos") is not None:
                px, py = int(data["field_pos"][0]*scale), int(data["field_pos"][1]*scale)
                if 0 <= px <= 1050 and 0 <= py <= 680:
                    cv2.circle(m, (px, py), 8 if cat != "ball" else 5, col, -1)
    return m

def draw_combined_view(frame, tracks, kpts, f_idx, tracker):
    # Draw field geometry lines
    if kpts:
        for s, e in FIELD_CONNECTIONS:
            if s in kpts and e in kpts:
                cv2.line(frame, tuple(map(int, kpts[s])), tuple(map(int, kpts[e])), (0, 0, 0), 2)

    # Draw player bounding boxes
    annotated = tracker.draw_annotations([frame], {
        k: [v[f_idx]] for k, v in tracks.items() if k != "keypoints"
    })[0]
    
    # Draw coordinate labels
    for cat in ["players", "goalkeepers"]:
        for tid, data in tracks[cat][f_idx].items():
            if data.get("field_pos") is not None:
                fx, fy = data["field_pos"]
                lx, ly = int((data["bbox"][0] + data["bbox"][2]) / 2), int(data["bbox"][1]) - 10
                label = f"({fx:.1f}, {fy:.1f})m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (lx - tw//2 - 2, ly - th - 2), (lx + tw//2 + 2, ly + 2), (0, 0, 0), -1)
                cv2.putText(annotated, label, (lx - tw//2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Overlay minimap
    m_map = draw_minimap(tracks, f_idx)
    m_small = cv2.resize(m_map, (350, 227)) 
    h, w = annotated.shape[:2]
    annotated[20:247, w-370:w-20] = m_small
    cv2.rectangle(annotated, (w-370, 20), (w-20, 247), (255, 255, 255), 2)
    return annotated

# ==============================================================================
# 4. MAIN EXECUTION
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

        homography_engine = StableHomography()
        
        # Initialize Physics Filters
        filters = {c: PlayerPositionFilter(max_movement_per_frame=2.0) for c in ["players", "goalkeepers", "referees"]}
        filters["ball"] = PlayerPositionFilter(max_movement_per_frame=4.0) # Ball moves faster

        final_tracks = {c: [] for c in ["players", "goalkeepers", "referees", "ball"]}

        for f_idx in range(video.total_frames):
            H = homography_engine.update(all_kpts[f_idx])

            for cat in final_tracks:
                frame_data = {}
                for tid, d in p_tracks[cat][f_idx].items():
                    foot = get_player_foot_position(d["bbox"])
                    raw_pos = transform_to_field_coords(foot, H)
                    
                    # Apply the Physics Filter to catch teleports
                    safe_pos = filters[cat].update(tid, raw_pos)
                    
                    frame_data[tid] = {"bbox": d["bbox"], "field_pos": safe_pos}
                final_tracks[cat].append(frame_data)

        writer = VideoWriter('output_videos/full_tracking_final.mp4', video.width, video.height, video.fps)

        video.reset()
        for f_idx, (_, batch) in enumerate(video.get_batch_generator(batch_size=1)):
            # Restore the rendering function call!
            annotated_frame = draw_combined_view(batch[0], final_tracks, all_kpts[f_idx], f_idx, p_tracker)
            writer.write(annotated_frame)

        writer.release()
        print("✅ SUCCESS: Physics filters applied and video rendered.")

if __name__ == '__main__':
    main()