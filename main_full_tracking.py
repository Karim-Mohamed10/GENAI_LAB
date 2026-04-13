import torch
import ultralytics
from utils.video import VideoProcessor, VideoWriter
from tracker.tracker import Tracker
from tracker.keypoints_tracker import KeypointsTracker
from Camera_estimator.Cam_Estimator import Cam_Estimator
from TeamFeatures.team_assigner import TeamAssigner
from TeamFeatures.caeAssigner import CAETeamAssigner
from TeamFeatures.possession_tracker import PossessionTracker
from TeamFeatures.speed_estimator import SpeedEstimator
from TeamFeatures.pass_detector import PassDetector
from TeamFeatures.goalkeeper_detector import GoalkeeperDetector
from TeamFeatures.tackle_detector import TackleDetector
from TeamFeatures.Card_Detector import CardDetector
import cv2
import numpy as np
import imageio
import os
import json
import sys
import draw_pass_maps

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
        self.smoothed_kpts = {}  
        self.alpha = alpha      

    def update(self, detected_keypoints):
        
        smoothed_current = {}
        for kid, coords in detected_keypoints.items():
            if kid in self.smoothed_kpts:
               
                smoothed_current[kid] = (
                    self.alpha * np.array(coords, dtype=np.float32) + 
                    (1 - self.alpha) * self.smoothed_kpts[kid]
                )
            else:
                smoothed_current[kid] = np.array(coords, dtype=np.float32)
            
            self.smoothed_kpts[kid] = smoothed_current[kid]

        src_pts, dst_pts = [], []
        for kid, coords in smoothed_current.items():
            if kid in STANDARD_FIELD_COORDS:
                src_pts.append(coords)
                dst_pts.append(STANDARD_FIELD_COORDS[kid])
        
        if len(src_pts) < 4:
            return self.last_valid_H
            
        src_arr = np.array(src_pts, dtype=np.float32)
        dst_arr = np.array(dst_pts, dtype=np.float32)
        
        x, y, w, h = cv2.boundingRect(src_arr)
        if w < 50 or h < 50: 
            return self.last_valid_H 
            
        H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 2.0)
        
        if H is not None and abs(np.linalg.det(H)) > 1e-6:
            self.last_valid_H = H
            
        return self.last_valid_H


class PlayerEMAFilter:
    def __init__(self, alpha=0.3, max_dist=3.0):
        self.positions = {}
        self.alpha = alpha
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
                
                if data.get("has_ball"):
                    col = (255, 0, 255) # Magenta in BGR
                else:
                    col = team_colors_map.get(t_id, (0, 255, 0))
                
                cv2.circle(m, (px, py), 12, col, -1)
                
                # Add a white ring for extra visibility
                if data.get("has_ball"):
                    cv2.circle(m, (px, py), 16, (255, 255, 255), 2)
    # Goalkeepers – default colour (green) with white border, no team colour
    for tid, data in tracks["goalkeepers"][frame_index].items():
        if data.get("field_pos") is not None:
            px, py = int(data["field_pos"][0]*scale), int(data["field_pos"][1]*scale)
            if -50 <= px <= 1100 and -50 <= py <= 730:
                cv2.circle(m, (px, py), 14, (255, 255, 255), -1)
                cv2.circle(m, (px, py), 12, (0, 255, 0), -1)

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
    
    # --- DEBUG: Highlight player with the ball ---
    for tid, data in tracks["players"][f_idx].items():
        if data.get("has_ball") and data.get("bbox") is not None:
            x1, y1, x2, y2 = [int(v) for v in data["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv2.putText(annotated, f"BALL (P{tid})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

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


def get_closest_player_id(ball_position, players_frame, team_id):
    if ball_position is None or team_id not in (1, 2):
        return None

    closest_player_id = None
    closest_distance = float("inf")
    ball_point = np.array(ball_position, dtype=np.float32)

    for player_id, player_data in players_frame.items():
        if player_data.get("team") != team_id:
            continue
        player_position = player_data.get("field_pos")
        if player_position is None:
            continue

        distance = float(np.linalg.norm(np.array(player_position, dtype=np.float32) - ball_point))
        if distance < closest_distance:
            closest_distance = distance
            closest_player_id = player_id

    return closest_player_id


def get_ball_possessor(ball_position, players_frame, max_possession_distance=2.3):
    if ball_position is None:
        return None, None

    ball_point = np.array(ball_position, dtype=np.float32)
    best_id = None
    best_team = None
    best_dist = float("inf")

    for player_id, player_data in players_frame.items():
        pos = player_data.get("field_pos")
        team = player_data.get("team")
        if pos is None or team not in (1, 2):
            continue

        dist = float(np.linalg.norm(np.array(pos, dtype=np.float32) - ball_point))
        if dist < best_dist:
            best_dist = dist
            best_id = player_id
            best_team = team

    if best_id is None or best_dist > max_possession_distance:
        return None, None

    return best_id, best_team


def get_nearest_player_team(target_position, players_frame):
    if target_position is None:
        return None

    player_distances = []
    target_point = np.array(target_position, dtype=np.float32)

    for _, player_data in players_frame.items():
        player_position = player_data.get("field_pos")
        player_team = player_data.get("team")
        if player_position is None or player_team not in (1, 2):
            continue

        distance = float(np.linalg.norm(np.array(player_position, dtype=np.float32) - target_point))
        player_distances.append((distance, player_team))

    if not player_distances:
        return None

    player_distances.sort(key=lambda x: x[0])
    nearest_three = player_distances[:3]

    team_counts = {1: 0, 2: 0}
    team_distance_sums = {1: 0.0, 2: 0.0}
    for dist, team in nearest_three:
        team_counts[team] += 1
        team_distance_sums[team] += dist

    if team_counts[1] > team_counts[2]:
        return 1
    if team_counts[2] > team_counts[1]:
        return 2

    return 1 if team_distance_sums[1] <= team_distance_sums[2] else 2


def assign_goalkeeper_team_from_position(field_pos, field_center_x=52.5):
    if field_pos is None:
        return None
    return 1 if float(field_pos[0]) < field_center_x else 2


def get_team_assigner_choice():
    """Prompt user to choose between TeamAssigner and CAETeamAssigner at runtime."""
    print("\n" + "="*60)
    print("TEAM ASSIGNMENT METHOD SELECTION")
    print("="*60)
    print("1. TeamAssigner (Color-based, traditional approach)")
    print("2. CAETeamAssigner (Deep learning with CAE embeddings)")
    print("="*60)
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")


def main():
    assigner_choice = get_team_assigner_choice()
    
    with VideoProcessor('input_videos/CHEvsMCI.mp4') as video:
        frames = []
        for _, batch in video.get_batch_generator(batch_size=1):
            frames.append(batch[0])

        if not frames:
            raise ValueError('No frames were loaded from the input video')

        cam_estimator = Cam_Estimator(frames[0])
        camera_movement_per_frame = cam_estimator.get_camera_movement(frames)

        p_tracker = Tracker('models/best.pt')
        k_tracker = KeypointsTracker('models/Keypoints_detection_best.pt', conf=0.3, kp_conf=0.5)

        def frame_gen():
            for frame in frames:
                yield frame

        p_tracks = p_tracker.get_object_tracks_chunked(frame_gen(), chunk_size=50)

        all_kpts = []
        for start_idx in range(0, len(frames), 10):
            batch = frames[start_idx:start_idx + 10]
            dets = k_tracker.detect(batch)
            for d in dets:
                all_kpts.append(k_tracker.track(d))

        # --- Team Assignment ---
        if assigner_choice == 1:
            team_assigner = TeamAssigner()
            assigner_type = "TeamAssigner (Color-based)"
            print(f"\n✓ Using {assigner_type}")
        else:
            team_assigner = CAETeamAssigner()
            assigner_type = "CAETeamAssigner (Deep Learning)"
            print(f"\n✓ Using {assigner_type}")

        if isinstance(team_assigner, CAETeamAssigner):
            bootstrap_ok = team_assigner.bootstrap_from_video(
                frames,
                p_tracks["players"],
                sample_every=5,
                max_frames=min(150, len(frames)),
            )

            if bootstrap_ok:
                print(f"Team assignment initialized using CAE bootstrap in {team_assigner.assignment_mode} mode")
            else:
                print("WARNING: CAE bootstrap failed, falling back to color-based bootstrap")
                all_player_colors = []
                max_init_frames = min(150, len(frames))
                for batch_idx, frame in enumerate(frames[:max_init_frames]):
                    if batch_idx % 5 != 0:
                        continue
                    player_dets = p_tracks["players"][batch_idx]
                    for _, det in player_dets.items():
                        color = team_assigner.get_player_color(frame, det["bbox"])
                        if color is not None:
                            all_player_colors.append(color)

                if len(all_player_colors) >= 2:
                    team_assigner.assign_team_color_from_colors(all_player_colors)
                    print(f"Team assignment initialized with {len(all_player_colors)} player color samples")
                else:
                    print("WARNING: Could not initialize team assignment - not enough player detections")
        else:
            all_player_colors = []
            max_init_frames = min(150, len(frames))
            for batch_idx, frame in enumerate(frames[:max_init_frames]):
                if batch_idx % 5 != 0:
                    continue
                player_dets = p_tracks["players"][batch_idx]
                for _, det in player_dets.items():
                    color = team_assigner.get_player_color(frame, det["bbox"])
                    if color is not None:
                        all_player_colors.append(color)

            if len(all_player_colors) >= 2:
                team_assigner.assign_team_color_from_colors(all_player_colors)
                print(f"Team assignment initialized with {len(all_player_colors)} player color samples")
            else:
                print("WARNING: Could not initialize team assignment - not enough player detections")

        if 1 in team_assigner.team_colors and 2 in team_assigner.team_colors:
            print(f"Team 1 color: {team_assigner.team_colors[1]}")
            print(f"Team 2 color: {team_assigner.team_colors[2]}")
        else:
            print("WARNING: Team colors were not initialized")

        view_transformer = ViewTransformer(alpha=0.5)
        possession_tracker = PossessionTracker(possession_radius=1.2, smoothing_window=5)
        # field_pos is already in real-world metres so scale factors = 1.0
        speed_estimator = SpeedEstimator(field_width=105, field_height=68,
                                         real_field_length=105.0, real_field_width=68.0,
                                         smoothing_window=5)

        # FIXED: Give the players a 3.0m speed limit, but give the ball an infinite (100.0m) limit so it never freezes
        filters = {c: PlayerEMAFilter(alpha=0.25, max_dist=3.0) for c in ["players", "goalkeepers", "referees"]}
        filters["ball"] = PlayerEMAFilter(alpha=1, max_dist=100.0) 

        final_tracks = {c: [] for c in ["players", "goalkeepers", "referees", "ball"]}

        KNOWN_MARKS = [(11.0, 34.0), (94.0, 34.0), (52.5, 34.0)] # penalty spots and center

        pass_detector = PassDetector()
        gk_detector = GoalkeeperDetector()
        tackle_detector = TackleDetector(fps=video.fps)
        card_detector = CardDetector(model_path='models/cards.pt')
        match_passes = []
        match_tackles = []
        match_cards = []
        for f_idx in range(video.total_frames):
            cur_frame = frames[f_idx]
            camera_shift = camera_movement_per_frame[f_idx] if f_idx < len(camera_movement_per_frame) else (0, 0)

            player_team_ids = {}
            if isinstance(team_assigner, CAETeamAssigner):
                player_team_ids = team_assigner.assign_teams_for_frame(
                    cur_frame,
                    p_tracks["players"][f_idx],
                    frame_idx=f_idx,
                )

            H = view_transformer.update(all_kpts[f_idx])

            # --- Extract Goalkeepers dynamically ---
            extracted_gks = gk_detector.separate_goalkeepers(
                p_tracks["players"][f_idx], 
                H, 
                get_foot_position, 
                transform_to_field_coords
            )
            # Add them safely to the goalkeepers track dictionary
            if extracted_gks:
                p_tracks["goalkeepers"][f_idx].update(extracted_gks)

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

                    # No need to adjust with optical flow pixel shifts.
                    safe_pos = filters[cat].update(tid, raw_pos)
                    
                    entry = {"bbox": d["bbox"], "field_pos": safe_pos}

                    # Assign team for players only (goalkeepers keep default colour)
                    if cat == "players" and team_assigner.kmeans is not None:
                        if isinstance(team_assigner, CAETeamAssigner):
                            team_id = player_team_ids.get(tid, 1)
                        else:
                            team_id = team_assigner.get_player_team(cur_frame, d["bbox"], tid)
                        entry["team"] = team_id

                    if cat == "goalkeepers":
                        entry["team"] = assign_goalkeeper_team_from_position(safe_pos)

                    # Provide projection (metres) for speed estimation
                    if safe_pos is not None and cat in ("players", "goalkeepers", "ball"):
                        entry["projection"] = (float(safe_pos[0]), float(safe_pos[1]))

                    frame_data[tid] = entry
                final_tracks[cat].append(frame_data)

            # --- Speed estimation for this frame ---
            speed_estimator.calculate_speed(
                {"players": final_tracks["players"][-1],
                 "goalkeepers": final_tracks["goalkeepers"][-1],
                 "ball": final_tracks["ball"][-1]},
                f_idx, video.fps
            )

            # --- Extract ball info ---
            ball_pos = None
            ball_speed = None
            for _, bd in final_tracks["ball"][-1].items():
                ball_pos = bd.get("field_pos")
                ball_speed = bd.get("speed")
                break

            # --- Pass detection for this frame ---
            pass_event = pass_detector.update(
                ball_pos, ball_speed, final_tracks["players"][-1], f_idx, video.fps
            )

            # --- DEBUG: Tag the player the system thinks has the ball ---
            if ball_pos is not None:
                closest_id, _, best_dist = pass_detector._get_closest_player(ball_pos, final_tracks["players"][-1])
                if closest_id is not None and best_dist <= pass_detector.possession_radius:
                    if closest_id in final_tracks["players"][-1]:
                        final_tracks["players"][-1][closest_id]["has_ball"] = True
            if pass_event is not None:
                match_passes.append(pass_event)
                print(f"SUCCESS: Pass event detected: {pass_event}")

            # --- Possession update for this frame ---
            possession_team = possession_tracker.update(ball_pos, final_tracks["players"][-1])

            # --- Tackle detection for this frame ---
            # Prefer direct ball possessor from geometry, fallback to smoothed team estimate.
            possessing_player_id, possessing_team_id = get_ball_possessor(ball_pos, final_tracks["players"][-1])
            if possessing_player_id is None:
                possessing_team_id = possession_team if possession_team in (1, 2) else None
                if possessing_team_id in (1, 2):
                    possessing_player_id = get_closest_player_id(ball_pos, final_tracks["players"][-1], possessing_team_id)
            
            tackle_events = tackle_detector.update(
                frame_idx=f_idx,
                players_frame=final_tracks["players"][-1],
                possessing_player_id=possessing_player_id,
                possessing_team_id=possessing_team_id,
                frame_image=cur_frame,
                ball_position=ball_pos,
            )
            for event in tackle_events:
                match_tackles.append(event)
                print(f"TACKLE: {event['status']} - Tackler P{event['tackler_id']} vs Victim P{event['victim_id']}")

            # --- Card detection for this frame ---
            players_for_cards = {}
            for pid, pdata in final_tracks["players"][-1].items():
                players_for_cards[pid] = {
                    "bbox": pdata.get("bbox"),
                    "team": pdata.get("team"),
                }

            card_events = card_detector.update(
                frame=cur_frame,
                players_in_frame=players_for_cards,
                frame_index=f_idx,
            )
            for event in card_events:
                match_cards.append(event)
                print(
                    f"CARD: {event['card_type'].upper()} for P{event['player_id']} "
                    f"(Team {event.get('team')})"
                )

        writer = VideoWriter('output_videos/LIVvsRMAnew.mp4', video.width, video.height, video.fps)

        # Build BGR team colours for the possession bar (kmeans clusters are BGR)
        if team_assigner.kmeans is not None:
            poss_colors = {
                1: tuple(int(c) for c in team_assigner.team_colors[1]),
                2: tuple(int(c) for c in team_assigner.team_colors[2]),
            }
        else:
            poss_colors = None

        for f_idx, frame in enumerate(frames):
            annotated_frame = draw_combined_view(frame, final_tracks, all_kpts[f_idx], f_idx, p_tracker, team_colors=poss_colors)
            annotated_frame = possession_tracker.draw_possession_bar_at(
                annotated_frame, f_idx, team_colors=poss_colors
            )

            # --- DEBUG: Pass Visualization Overlay ---
            active_pass = None
            for p in match_passes:
                start_f = p.get("start_frame", 0)
                end_f = p.get("end_frame", 0)
                if start_f <= f_idx <= end_f + int(video.fps * 1.5):
                    active_pass = p
                    break
            
            if active_pass:
                text = f"{active_pass['status']} PASS: P{active_pass['initiator_id']} -> P{active_pass['receiver_id']}"
                bg_color = (0, 0, 0)
                text_color = (0, 255, 0) if active_pass['status'] == "COMPLETED" else (0, 0, 255)
                
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(annotated_frame, (40, 40), (40 + tw + 20, 40 + th + 30), bg_color, -1)
                cv2.putText(annotated_frame, text, (50, 40 + th + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)

            # --- Tackle Visualization Overlay ---
            active_tackle = None
            for t in match_tackles:
                start_f = t.get("start_frame", 0)
                end_f = t.get("end_frame", 0)
                if start_f <= f_idx <= end_f + int(video.fps * 1.5):
                    active_tackle = t
                    break
            
            if active_tackle:
                status_text = "SUCCESSFUL" if active_tackle['outcome'] == "success" else "FAILED"
                foul_text = ""
                if "is_foul_model" in active_tackle:
                    foul_tag = "FOUL" if active_tackle["is_foul_model"] else "NO FOUL"
                    foul_prob = active_tackle.get("foul_probability", 0.0)
                    foul_text = f" | {foul_tag} ({foul_prob:.2f})"
                text = f"{status_text} TACKLE: P{active_tackle['tackler_id']} vs P{active_tackle['victim_id']}{foul_text}"
                bg_color = (0, 0, 0)
                text_color = (0, 255, 0) if active_tackle['outcome'] == "success" else (0, 0, 255)
                
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                # Position below the pass overlay to avoid overlap
                y_offset = 40 if not active_pass else 120
                cv2.rectangle(annotated_frame, (40, y_offset), (40 + tw + 20, y_offset + th + 30), bg_color, -1)
                cv2.putText(annotated_frame, text, (50, y_offset + th + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)

            # --- Card Visualization Overlay ---
            active_card = None
            for c in match_cards:
                if c.get("frame") == f_idx:
                    active_card = c
                    break

            if active_card:
                card_type = active_card.get("card_type", "").upper()
                text = f"{card_type} CARD: P{active_card['player_id']}"
                if active_card.get("team") in (1, 2):
                    text += f" (Team {active_card['team']})"

                bg_color = (0, 0, 0)
                text_color = (0, 255, 255) if card_type == "YELLOW" else (0, 0, 255)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

                y_offset = 200
                if active_pass and active_tackle:
                    y_offset = 280
                elif active_pass or active_tackle:
                    y_offset = 200

                cv2.rectangle(annotated_frame, (40, y_offset), (40 + tw + 20, y_offset + th + 26), bg_color, -1)
                cv2.putText(annotated_frame, text, (50, y_offset + th + 13), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            
            writer.write(annotated_frame)

        writer.release()
        
        # --- Save Passes to JSON ---
        with open('passes.json', 'w') as f:
            json.dump(match_passes, f, indent=4)
        print(f"Successfully saved {len(match_passes)} pass events to passes.json")

        # --- Save Tackles to JSON ---
        with open('tackles.json', 'w') as f:
            json.dump(match_tackles, f, indent=4)
        print(f"Successfully saved {len(match_tackles)} tackle events to tackles.json")

        # --- Save Cards to JSON ---
        cards_output = {
            "events": match_cards,
            "summary": card_detector.get_summary(),
        }
        with open('cards.json', 'w') as f:
            json.dump(cards_output, f, indent=4)
        print(f"Successfully saved {len(match_cards)} card events to cards.json")

        t1_pct, t2_pct = possession_tracker.get_possession_percentages()
        print(f"SUCCESS: Ball now tracks properly on minimap and displays coordinates!")
        print(f"Final Possession  —  Team 1: {t1_pct}%  |  Team 2: {t2_pct}%")
        
        # --- Generate Pass Maps ---
        print("Generating pass maps")
        draw_pass_maps.main()

if __name__ == '__main__':
    main()