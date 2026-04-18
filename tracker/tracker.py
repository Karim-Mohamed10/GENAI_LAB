from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

class Tracker:
    def __init__(self, modelPath, conf=0.5, ball_conf=0.3):
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()
        self.conf = conf
        self.ball_conf = ball_conf

        # Ball temporal state used to avoid shoe hijacking and reduce dropouts.
        self.ball_state = {
            "last_bbox": None,
            "last_center": None,
            "velocity": np.array([0.0, 0.0], dtype=np.float32),
            "missed": 0,
        }
        self.max_ball_jump_px = 120.0
        self.max_ball_missed_frames = 4
        self.max_ball_area_ratio = 0.006
        self.min_ball_area_px = 8.0
        
    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            batch=self.model.predict(frames[i:i+batch_size], conf=self.conf, verbose=False, show=False)
            detections+=batch
        return detections

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        union = self._bbox_area(box_a) + self._bbox_area(box_b) - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def _valid_ball_geometry(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area = bw * bh
        frame_area = float(max(1, frame_w * frame_h))

        # The ball should be small and roughly compact in image space.
        if area < self.min_ball_area_px:
            return False
        if area > frame_area * self.max_ball_area_ratio:
            return False

        aspect = bw / bh
        if aspect < 0.35 or aspect > 2.8:
            return False
        return True

    def _update_ball_state(self, bbox):
        center = self._bbox_center(bbox)
        last_center = self.ball_state["last_center"]
        if last_center is not None:
            self.ball_state["velocity"] = center - last_center
        self.ball_state["last_bbox"] = [float(v) for v in bbox]
        self.ball_state["last_center"] = center
        self.ball_state["missed"] = 0

    def _predict_ball_bbox(self):
        last_bbox = self.ball_state["last_bbox"]
        last_center = self.ball_state["last_center"]
        velocity = self.ball_state["velocity"]
        if last_bbox is None or last_center is None:
            return None

        pred_center = last_center + velocity
        x1, y1, x2, y2 = last_bbox
        w = x2 - x1
        h = y2 - y1
        return [
            float(pred_center[0] - w * 0.5),
            float(pred_center[1] - h * 0.5),
            float(pred_center[0] + w * 0.5),
            float(pred_center[1] + h * 0.5),
        ]

    def _select_ball_bbox(self, detection_supervision, cls_names_inv, occluder_bboxes, frame_w, frame_h):
        ball_ids = set(cls_names_inv.get('ball', []))
        if not ball_ids:
            self.ball_state["missed"] += 1
            return None

        predicted_bbox = self._predict_ball_bbox()
        predicted_center = self._bbox_center(predicted_bbox) if predicted_bbox is not None else None
        frame_diag = float(np.hypot(frame_w, frame_h)) if frame_w > 0 and frame_h > 0 else 1.0

        best_score = -1e9
        best_bbox = None

        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            conf = float(frame_detection[2])
            cls_id = int(frame_detection[3])

            if cls_id not in ball_ids:
                continue
            if conf < self.ball_conf:
                continue
            if not self._valid_ball_geometry(bbox, frame_w, frame_h):
                continue

            center = self._bbox_center(bbox)
            dist_score = 0.5
            if predicted_center is not None:
                jump = float(np.linalg.norm(center - predicted_center))
                if jump > self.max_ball_jump_px and conf < 0.6:
                    continue
                dist_score = max(0.0, 1.0 - (jump / max(1.0, frame_diag * 0.25)))

            max_iou = 0.0
            for occ_bbox in occluder_bboxes:
                max_iou = max(max_iou, self._bbox_iou(bbox, occ_bbox))

            conf_score = (conf - self.ball_conf) / max(1e-6, (1.0 - self.ball_conf))
            score = (0.62 * conf_score) + (0.38 * dist_score) - (0.55 * max_iou)

            if score > best_score:
                best_score = score
                best_bbox = bbox

        if best_bbox is not None:
            self._update_ball_state(best_bbox)
            return best_bbox

        self.ball_state["missed"] += 1
        if self.ball_state["missed"] <= self.max_ball_missed_frames:
            return self._predict_ball_bbox()

        # Drop stale state after a longer miss window.
        self.ball_state["last_bbox"] = None
        self.ball_state["last_center"] = None
        self.ball_state["velocity"] = np.array([0.0, 0.0], dtype=np.float32)
        return None
    
    def get_object_tracks(self, frames): #to be removed (Karim) the chunked version is more efficient
        detections = self.detect_frames(frames)
        
        tracks={
            "players":[],
            "goalkeepers":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            
            cls_names_inv = defaultdict(list) 
            for k,v in cls_names.items():
                cls_names_inv[v].append(k)

            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id in cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id in cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id in cls_names_inv['goalkeeper']:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}

            occluders = [
                d["bbox"] for d in tracks["players"][frame_num].values()
            ] + [
                d["bbox"] for d in tracks["goalkeepers"][frame_num].values()
            ]
            frame_h, frame_w = detection.orig_shape if hasattr(detection, "orig_shape") else (1080, 1920)
            best_ball_bbox = self._select_ball_bbox(
                detection_supervision,
                cls_names_inv,
                occluders,
                frame_w,
                frame_h,
            )

            if best_ball_bbox is not None:
                tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox}
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])
        return tracks
    

    def get_object_tracks_chunked(self, frame_generator, chunk_size=50):
        tracks={
            "players":[],
            "goalkeepers":[],
            "referees":[],
            "ball":[]
        }
        
        frame_num = 0
        chunk = []
        
        for frame in frame_generator:
            chunk.append(frame)
            
            if len(chunk) >= chunk_size:
                # Process this chunk
                detections = self.detect_frames(chunk)
                
                for detection in detections:
                    cls_names = detection.names
                    
                    cls_names_inv = defaultdict(list) 
                    for k,v in cls_names.items():
                        cls_names_inv[v].append(k)

                    detection_supervision = sv.Detections.from_ultralytics(detection)
                    detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
                    
                    tracks["players"].append({})
                    tracks["goalkeepers"].append({})
                    tracks["referees"].append({})
                    tracks["ball"].append({})

                    for frame_detection in detection_with_tracks:
                        bbox = frame_detection[0].tolist()
                        cls_id = frame_detection[3]
                        track_id = frame_detection[4]

                        if cls_id in cls_names_inv['player']:
                            tracks["players"][frame_num][track_id] = {"bbox":bbox}
                            
                        if cls_id in cls_names_inv['referee']:
                            tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                            
                        if cls_id in cls_names_inv['goalkeeper']:
                            tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}

                    occluders = [
                        d["bbox"] for d in tracks["players"][frame_num].values()
                    ] + [
                        d["bbox"] for d in tracks["goalkeepers"][frame_num].values()
                    ]
                    frame_h, frame_w = detection.orig_shape if hasattr(detection, "orig_shape") else (1080, 1920)
                    best_ball_bbox = self._select_ball_bbox(
                        detection_supervision,
                        cls_names_inv,
                        occluders,
                        frame_w,
                        frame_h,
                    )
                    
                    if best_ball_bbox is not None:
                        tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox}
                    
                    frame_num += 1
                
                # Clear chunk for next iteration
                chunk = []
                
        # Process remaining frames
        if chunk:
            detections = self.detect_frames(chunk)
            
            for detection in detections:
                cls_names = detection.names
                
                cls_names_inv = defaultdict(list) 
                for k,v in cls_names.items():
                    cls_names_inv[v].append(k)

                detection_supervision = sv.Detections.from_ultralytics(detection)
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
                
                tracks["players"].append({})
                tracks["goalkeepers"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id in cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox":bbox}
                        
                    if cls_id in cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                        
                    if cls_id in cls_names_inv['goalkeeper']:
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}

                occluders = [
                    d["bbox"] for d in tracks["players"][frame_num].values()
                ] + [
                    d["bbox"] for d in tracks["goalkeepers"][frame_num].values()
                ]
                frame_h, frame_w = detection.orig_shape if hasattr(detection, "orig_shape") else (1080, 1920)
                best_ball_bbox = self._select_ball_bbox(
                    detection_supervision,
                    cls_names_inv,
                    occluders,
                    frame_w,
                    frame_h,
                )
                
                if best_ball_bbox is not None:
                    tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox}
                
                frame_num += 1
                
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])
        return tracks

    def interpolate_ball_positions(self, ball_positions):
        bboxes = []
        for frame_ball in ball_positions:
            bbox = frame_ball.get(1, {}).get('bbox', [])
            if len(bbox) == 0:
                bboxes.append([np.nan, np.nan, np.nan, np.nan])
            else:
                bboxes.append(bbox)
        
        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        
        interpolated_positions = [{1: {"bbox": row}} if not np.isnan(row[0]) else {} for row in df.to_numpy().tolist()]
        return interpolated_positions


    def draw_player_indicators(self, frame, player_tracks, color=(0,255,0), use_team_color=True):
        # Team colour mapping: team 1 -> red (BGR), team 2 -> blue (BGR)
        team_colors_map = {1: (0, 0, 255), 2: (255, 0, 0)}

        for track_id, player in player_tracks.items():
            bbox = player["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            center_x = int((x1 + x2) / 2)
            width = x2 - x1
            
            # Determine Team Color
            draw_color = color
            if use_team_color and "team" in player:
                draw_color = team_colors_map.get(player["team"], color)

            # ---------------------------------------------------------
            # MODERN DESIGN: 3D Floor Ring (Broadcast Standard)
            # ---------------------------------------------------------
            # Create a flattened circle (ellipse) based on player width
            # Height of ellipse is 1/6th of the width to give a 3D perspective
            axes = (int(width / 2), int(width / 6)) 
            center = (center_x, y2) # Anchored exactly at the heels
            
            # 1. Draw a thick black shadow/outline first
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 0), 4)
            # 2. Draw the thinner team-colored ring inside the shadow
            cv2.ellipse(frame, center, axes, 0, 0, 360, draw_color, 2)

            # ---------------------------------------------------------
            # CLEAN ID TAG: Ghost Text Floating Above Head
            # ---------------------------------------------------------
            text = str(track_id)
            
            # Calculate text size to center it perfectly above the head
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = center_x - (tw // 2)
            text_y = y1 - 10 # 10 pixels above the bounding box
            
            # Draw thick black text first (acts as an outline/shadow)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            # Draw the bright white text exactly on top
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def draw_annotations(self, video_frames, tracks):

        output_video_frames = []

        for frame_index, frame in enumerate(video_frames):

            player_hashmap = tracks["players"][frame_index]
            referee_hashmap = tracks["referees"][frame_index]
            ball_hashmap = tracks["ball"][frame_index]
            goalkeeper_hashmap = tracks["goalkeepers"][frame_index]

            # Use team colours for players; goalkeepers always draw in neutral yellow
            frame = self.draw_player_indicators(frame, player_hashmap, (0,255,0), use_team_color=True)

            frame = self.draw_player_indicators(frame, goalkeeper_hashmap, (0, 255, 0), use_team_color=False)

            frame = self.draw_player_indicators(frame, referee_hashmap, (0,255,255), use_team_color=False)
            
            frame = self.draw_player_indicators(frame, ball_hashmap, (255,255,255), use_team_color=False)

            output_video_frames.append(frame)

        return output_video_frames
