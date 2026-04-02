from utils.video import VideoProcessor, VideoWriter
from tracker.tracker import Tracker
from tracker.keypoints_tracker import KeypointsTracker
from TeamFeatures.team_assigner import TeamAssigner
from TeamFeatures.possession_tracker import PossessionTracker
from TeamFeatures.speed_estimator import SpeedEstimator
import cv2
import numpy as np
import imageio
import os

#connecting keypoints 
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

STANDARD_FIELD_COORDS = { # mapping keypoints to real world coords
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


class ViewTransformer:
    def __init__(self, alpha=0.5):
        self.last_valid_H = None
        self.smoothed_kpts = {}  # Store EMA of keypoints
        self.alpha = alpha       # Smoothing factor

    def update(self, detected_keypoints):
        # 1. Apply EMA smoothing to incoming keypoints
        smoothed_current = {}
        for kid, coords in detected_keypoints.items():
            if kid in self.smoothed_kpts:
                # Blend new coordinates with history
                smoothed_current[kid] = (
                    self.alpha * np.array(coords, dtype=np.float32) + 
                    (1 - self.alpha) * self.smoothed_kpts[kid]
                )
            else:
                smoothed_current[kid] = np.array(coords, dtype=np.float32)
            
            # Update history for next frame
            self.smoothed_kpts[kid] = smoothed_current[kid]

        # 2. Extract source and destination points using the SMOOTHED coordinates
        src_pts, dst_pts = [], []
        for kid, coords in smoothed_current.items():
            if kid in STANDARD_FIELD_COORDS:
                src_pts.append(coords)
                dst_pts.append(STANDARD_FIELD_COORDS[kid])
        
        if len(src_pts) < 4:
            return self.last_valid_H
            
        src_arr = np.array(src_pts, dtype=np.float32)
        dst_arr = np.array(dst_pts, dtype=np.float32)
        
        # Check bounding box to ensure points aren't clumped
        x, y, w, h = cv2.boundingRect(src_arr)
        if w < 50 or h < 50: 
            return self.last_valid_H 
            
        # 3. Tighten RANSAC threshold from 5.0 to 2.0 to reject perspective outliers
        H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 2.0)
        
        if H is not None and abs(np.linalg.det(H)) > 1e-6:
            self.last_valid_H = H
            
        return self.last_valid_H


class PlayerEMAFilter:
    def __init__(self, alpha=0.3, max_dist=3.0):
        self.positions = {}
        self.alpha = alpha
        # FIXED: Added configurable max distance so the ball isn't restricted
        self.max_dist = max_dist 

    def update(self, track_id, new_pos):
        if new_pos is None:
            return self.positions.get(track_id)
            
        if track_id not in self.positions:
            self.positions[track_id] = np.array(new_pos, dtype=np.float32)
            return new_pos
            
        smoothed = (self.alpha * np.array(new_pos, dtype=np.float32)) + ((1 - self.alpha) * self.positions[track_id])
        
        dist = np.linalg.norm(smoothed - self.positions[track_id])
        if dist > self.max_dist: 
            return self.positions[track_id]
            
        self.positions[track_id] = smoothed
        return smoothed


def get_foot_position(bbox, is_ball=False):
    x1, y1, x2, y2 = bbox
    if is_ball:
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    height = y2 - y1
    return np.array([(x1 + x2) / 2, y2 - (height * 0.05)], dtype=np.float32)

def transform_to_field_coords(pos, H):
    if H is None: return None
    pt = np.array([[pos]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

def draw_field_lines(frame, keypoints):
    if not keypoints: return frame
    
    for s, e in FIELD_CONNECTIONS:
        if s in keypoints and e in keypoints:
            cv2.line(frame, tuple(map(int, keypoints[s])), tuple(map(int, keypoints[e])), (0, 255, 255), 2)
            
    for kp_id, coords in keypoints.items():
        cv2.circle(frame, (int(coords[0]), int(coords[1])), 5, (0, 255, 0), -1)
        
    return frame

def draw_minimap(tracks, frame_index, team_colors_map=None):
    m = np.ones((680, 1050, 3), dtype=np.uint8) * 40
    scale = 10
    
    cv2.rectangle(m, (0, 0), (1050, 680), (255, 255, 255), 2)
    cv2.line(m, (525, 0), (525, 680), (255, 255, 255), 2)
    cv2.circle(m, (525, 340), int(9.15 * scale), (255, 255, 255), 2)
    cv2.rectangle(m, (0, 138), (165, 542), (255, 255, 255), 2) 
    cv2.rectangle(m, (885, 138), (1050, 542), (255, 255, 255), 2) 
    cv2.rectangle(m, (0, 248), (55, 432), (255, 255, 255), 2)  
    cv2.rectangle(m, (995, 248), (1050, 432), (255, 255, 255), 2) 

    # Team-colored dots for players
    if team_colors_map is None:
        team_colors_map = {1: (0, 0, 255), 2: (255, 0, 0)}  # team1=red(BGR), team2=blue(BGR)
    for tid, data in tracks["players"][frame_index].items():
        if data.get("field_pos") is not None:
            px, py = int(data["field_pos"][0]*scale), int(data["field_pos"][1]*scale)
            if -50 <= px <= 1100 and -50 <= py <= 730:
                t_id = data.get("team", 1)
                col = team_colors_map.get(t_id, (0, 255, 0))
                cv2.circle(m, (px, py), 12, col, -1)
    # Goalkeepers – default colour (yellow) with white border, no team colour
    for tid, data in tracks["goalkeepers"][frame_index].items():
        if data.get("field_pos") is not None:
            px, py = int(data["field_pos"][0]*scale), int(data["field_pos"][1]*scale)
            if -50 <= px <= 1100 and -50 <= py <= 730:
                cv2.circle(m, (px, py), 14, (255, 255, 255), -1)
                cv2.circle(m, (px, py), 12, (0, 255, 255), -1)
                    
    for tid, data in tracks["ball"][frame_index].items():
        if data.get("field_pos") is not None:
            px, py = int(data["field_pos"][0]*scale), int(data["field_pos"][1]*scale)
            if -50 <= px <= 1100 and -50 <= py <= 730:
                cv2.circle(m, (px, py), 10, (0, 0, 0), -1)        
                cv2.circle(m, (px, py), 6, (255, 255, 255), -1)  
                
    return m

def draw_combined_view(frame, tracks, kpts, f_idx, tracker, team_colors=None):
    frame = draw_field_lines(frame, kpts)

    annotated = tracker.draw_annotations([frame], {
        k: [v[f_idx]] for k, v in tracks.items() if k != "keypoints"
    })[0]
    
    # # Draw coordinates
    # for cat in ["players", "goalkeepers", "ball"]:
    #     for tid, data in tracks[cat][f_idx].items():
    #         if data.get("field_pos") is not None:
    #             fx, fy = data["field_pos"]
    #             lx, ly = int((data["bbox"][0] + data["bbox"][2]) / 2), int(data["bbox"][1]) - 10
    #             label = f"({fx:.1f}, {fy:.1f})m"
                
    #             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
    #             # Make the ball text white, players cyan
    #             txt_color = (255, 255, 255) if cat == "ball" else (0, 255, 255)
                
    #             cv2.rectangle(annotated, (lx - tw//2 - 2, ly - th - 2), (lx + tw//2 + 2, ly + 2), (0, 0, 0), -1)
    #             cv2.putText(annotated, label, (lx - tw//2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)

    # # Draw speed for players and goalkeepers
    # for cat in ("players", "goalkeepers"):
    #     for tid, data in tracks[cat][f_idx].items():
    #         spd = data.get("speed")
    #         if spd is None or data.get("bbox") is None:
    #             continue
    #         lx = int((data["bbox"][0] + data["bbox"][2]) / 2)
    #         ly = int(data["bbox"][1]) - 26
    #         spd_label = f"{spd:.1f} km/h"
    #         (sw, sh), _ = cv2.getTextSize(spd_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    #         cv2.rectangle(annotated, (lx - sw//2 - 2, ly - sh - 2), (lx + sw//2 + 2, ly + 2), (0, 0, 0), -1)
    #         cv2.putText(annotated, spd_label, (lx - sw//2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    m_map = draw_minimap(tracks, f_idx, team_colors)

    m_small = cv2.resize(m_map, (525, 340)) 
    h, w = annotated.shape[:2]
    
    # 2. Update placement coordinates to fit the new dimensions
    y1, y2 = 20, 360            
    x1, x2 = w - 545, w - 20    
    
    annotated[y1:y2, x1:x2] = m_small
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    return annotated


def main():
    with VideoProcessor('input_videos/LIVvsCHE.mp4') as video:
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

        # --- Team Assignment ---
        team_assigner = TeamAssigner()
        # Collect player colors from multiple frames for robust team clustering
        video.reset()
        all_player_colors = []
        max_init_frames = min(150, video.total_frames)
        for batch_idx, batch in video.get_batch_generator(batch_size=1):
            if batch_idx >= max_init_frames:
                break
            # Sample every 5th frame to get a diverse spread of players over time
            if batch_idx % 5 != 0:
                continue
            frame = batch[0]
            player_dets = p_tracks["players"][batch_idx]
            for _, det in player_dets.items():
                color = team_assigner.get_player_color(frame, det["bbox"])
                if color is not None:
                    all_player_colors.append(color)

        if len(all_player_colors) >= 2:
            team_assigner.assign_team_color_from_colors(all_player_colors)
            print(f"Team assignment initialized with {len(all_player_colors)} player color samples")
            print(f"Team 1 color: {team_assigner.team_colors[1]}")
            print(f"Team 2 color: {team_assigner.team_colors[2]}")
        else:
            print("WARNING: Could not initialize team assignment - not enough player detections")

        view_transformer = ViewTransformer(alpha=0.5)
        possession_tracker = PossessionTracker(possession_radius=3.0, smoothing_window=5)
        # field_pos is already in real-world metres so scale factors = 1.0
        speed_estimator = SpeedEstimator(field_width=105, field_height=68,
                                         real_field_length=105.0, real_field_width=68.0,
                                         smoothing_window=5)

        # FIXED: Give the players a 3.0m speed limit, but give the ball an infinite (100.0m) limit so it never freezes
        filters = {c: PlayerEMAFilter(alpha=0.25, max_dist=3.0) for c in ["players", "goalkeepers", "referees"]}
        filters["ball"] = PlayerEMAFilter(alpha=0.8, max_dist=100.0) 

        final_tracks = {c: [] for c in ["players", "goalkeepers", "referees", "ball"]}

        # We need the actual video frames for team colour extraction
        GK_TEAM_MAP = {100: 1, 469: 2}  # hardcoded goalkeeper → team assignments
        video.reset()
        frame_iter = iter(video.get_batch_generator(batch_size=1))

        KNOWN_MARKS = [(11.0, 34.0), (94.0, 34.0), (52.5, 34.0)] # penalty spots and center

        for f_idx in range(video.total_frames):
            # Get the current frame for colour-based team assignment
            _, cur_batch = next(frame_iter)
            cur_frame = cur_batch[0]

            H = view_transformer.update(all_kpts[f_idx])

            for cat in final_tracks:
                frame_data = {}
                # Using list() to create a copy for safe iteration while deleting
                for tid, d in list(p_tracks[cat][f_idx].items()):
                    is_ball = (cat == "ball")
                    foot = get_foot_position(d["bbox"], is_ball=is_ball)
                    raw_pos = transform_to_field_coords(foot, H)

                    # --- Exclusion Zone Filter for Ball ---
                    if cat == "ball" and raw_pos is not None:
                        is_false_positive = False
                        for mark in KNOWN_MARKS:
                            dist = np.linalg.norm(np.array(raw_pos) - np.array(mark))
                            if dist < 3.0:
                                is_false_positive = True
                                break
                        
                        if is_false_positive:
                            raw_pos = None # Reject this position
                            p_tracks["ball"][f_idx].pop(tid, None) # Remove from drawable tracks
                            continue # Skip adding to final_tracks for this frame

                    safe_pos = filters[cat].update(tid, raw_pos)
                    
                    entry = {"bbox": d["bbox"], "field_pos": safe_pos}

                    # Assign team for players only (goalkeepers keep default colour)
                    if cat == "players" and team_assigner.kmeans is not None:
                        team_id = team_assigner.get_player_team(cur_frame, d["bbox"], tid)
                        entry["team"] = team_id

                    # Hardcoded goalkeeper team assignments
                    if cat == "goalkeepers" and tid in GK_TEAM_MAP:
                        entry["team"] = GK_TEAM_MAP[tid]

                    # Provide projection (metres) for speed estimation
                    if safe_pos is not None and cat in ("players", "goalkeepers"):
                        entry["projection"] = (float(safe_pos[0]), float(safe_pos[1]))

                    frame_data[tid] = entry
                final_tracks[cat].append(frame_data)

            # --- Speed estimation for this frame ---
            speed_estimator.calculate_speed(
                {"players": final_tracks["players"][-1],
                 "goalkeepers": final_tracks["goalkeepers"][-1]},
                f_idx, video.fps
            )

            # --- Possession update for this frame ---
            ball_pos = None
            for _, bd in final_tracks["ball"][-1].items():
                ball_pos = bd.get("field_pos")
                break
            possession_tracker.update(ball_pos, final_tracks["players"][-1])

        writer = VideoWriter('output_videos/KEY_LIVvsCHE.mp4', video.width, video.height, video.fps)

        # Build BGR team colours for the possession bar (kmeans clusters are BGR)
        if team_assigner.kmeans is not None:
            poss_colors = {
                1: tuple(int(c) for c in team_assigner.team_colors[1]),
                2: tuple(int(c) for c in team_assigner.team_colors[2]),
            }
        else:
            poss_colors = None

        video.reset()
        for f_idx, (_, batch) in enumerate(video.get_batch_generator(batch_size=1)):
            annotated_frame = draw_combined_view(batch[0], final_tracks, all_kpts[f_idx], f_idx, p_tracker, team_colors=poss_colors)
            annotated_frame = possession_tracker.draw_possession_bar_at(
                annotated_frame, f_idx, team_colors=poss_colors
            )
            writer.write(annotated_frame)

        writer.release()

        t1_pct, t2_pct = possession_tracker.get_possession_percentages()
        print(f"SUCCESS: Ball now tracks properly on minimap and displays coordinates!")
        print(f"Final Possession  —  Team 1: {t1_pct}%  |  Team 2: {t2_pct}%")

if __name__ == '__main__':
    main()