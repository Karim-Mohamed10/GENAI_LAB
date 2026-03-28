from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from collections import defaultdict

class Tracker:
    def __init__(self,modelPath):
        self.model=YOLO(modelPath)
        self.tracker=sv.ByteTrack()
        
    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            batch=self.model.predict(frames[i:i+batch_size],conf=0.1,verbose=False, show=False)
            detections+=batch
        return detections
    
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

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == cls_names_inv['goalkeeper']:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}
                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
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

                        if cls_id == cls_names_inv['player']:
                            tracks["players"][frame_num][track_id] = {"bbox":bbox}
                            
                        if cls_id == cls_names_inv['referee']:
                            tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                            
                        if cls_id == cls_names_inv['goalkeeper']:
                            tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}
                        
                    for frame_detection in detection_supervision:
                        bbox = frame_detection[0].tolist()
                        cls_id = frame_detection[3]

                        if cls_id == cls_names_inv['ball']:
                            tracks["ball"][frame_num][1] = {"bbox":bbox}
                    
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

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox":bbox}
                        
                    if cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                        
                    if cls_id == cls_names_inv['goalkeeper']:
                        tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}
                    
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv['ball']:
                        tracks["ball"][frame_num][1] = {"bbox":bbox}
                
                frame_num += 1
        
        return tracks
    

    def draw_player_indicators(self, frame, player_tracks, color=(0,255,0), use_team_color=True):
        # Team colour mapping: team 1 -> red (BGR), team 2 -> blue (BGR)
        team_colors_map = {1: (0, 0, 255), 2: (255, 0, 0)}

        for track_id, player in player_tracks.items():
            bbox = player["bbox"]

            x1, y1, x2, y2 = map(int, bbox)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            radius = int(max(x2 - x1, y2 - y1) / 2)

            draw_color = color
            if use_team_color and "team" in player:
                draw_color = team_colors_map.get(player["team"], color)

            cv2.circle(frame, (center_x, center_y), radius, draw_color, 2)

            cv2.putText(frame, str(track_id),
                        (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, draw_color, 2)

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

            frame = self.draw_player_indicators(frame, goalkeeper_hashmap, (0, 255, 255), use_team_color=False)

            frame = self.draw_player_indicators(frame, referee_hashmap, (0,255,255), use_team_color=False)
            
            frame = self.draw_player_indicators(frame, ball_hashmap, (255,255,255), use_team_color=False)

            output_video_frames.append(frame)

        return output_video_frames
