import math

class PassDetector:
    def __init__(self, possession_radius=1.5, kick_speed_threshold=10.0, min_possession_frames=1):
        self.possession_radius = possession_radius
        self.kick_speed_threshold = kick_speed_threshold
        self.min_possession_frames = min_possession_frames
        
        # State variables
        self.state = "POSSESSION"
        self.current_possessor = None
        self.current_team = None
        self.possession_frames = 0
        
        # Pass tracking variables
        self.pass_start_pos = None
        self.initiator_id = None
        self.initiator_team = None
        self.pass_start_frame = None

    def _get_closest_player(self, ball_pos, players):
        """
        Calculates the Euclidean distance between the ball and each player's field_pos.
        
        Args:
            ball_pos: The (x, y) coordinates of the ball.
            players: A dict mapping player_id to player data (which includes 'field_pos' and 'team').
            
        Returns:
            tuple: (closest_id, closest_team, best_dist)
        """
        closest_id = None
        closest_team = None
        best_dist = float('inf')
        
        if ball_pos is None or not players:
            return closest_id, closest_team, best_dist
            
        bx, by = ball_pos
        
        for player_id, player_data in players.items():
            field_pos = player_data.get('field_pos')
            if field_pos is not None:
                px, py = field_pos
                dist = math.hypot(px - bx, py - by)
                
                if dist < best_dist:
                    best_dist = dist
                    closest_id = player_id
                    closest_team = player_data.get('team')
                    
        return closest_id, closest_team, best_dist

    def update(self, ball_pos, ball_speed, players, frame_idx, fps):
        closest_id, closest_team, best_dist = self._get_closest_player(ball_pos, players)

        if self.state == "POSSESSION":
            if best_dist <= self.possession_radius:
                self.current_possessor = closest_id
                self.current_team = closest_team
                self.possession_frames += 1
            elif ball_speed is not None and ball_speed > self.kick_speed_threshold and self.possession_frames >= self.min_possession_frames:
                self.state = "MOTION"
                self.initiator_id = self.current_possessor
                self.initiator_team = self.current_team
                self.pass_start_pos = ball_pos
                self.pass_start_frame = frame_idx
                self.current_possessor = None
                self.current_team = None
                self.possession_frames = 0
            else:
                self.current_possessor = None
                self.current_team = None
                self.possession_frames = 0

        elif self.state == "MOTION":
            
            # FIX 1: Pass received by a DIFFERENT player (Catches one-touch and fast passes)
            if best_dist <= self.possession_radius and closest_id is not None and closest_id != self.initiator_id:
                status = "COMPLETED" if closest_team == self.initiator_team else "INTERCEPTED"
                
                distance_meters = 0.0
                if self.pass_start_pos is not None and ball_pos is not None:
                    distance_meters = math.hypot(ball_pos[0] - self.pass_start_pos[0], ball_pos[1] - self.pass_start_pos[1])
                
                duration_seconds = 0.0
                if self.pass_start_frame is not None and fps:
                    duration_seconds = (frame_idx - self.pass_start_frame) / fps

                event = {
                    "event_type": "PASS",
                    "status": status,
                    "initiator_id": int(self.initiator_id) if self.initiator_id is not None else None,
                    "initiator_team": int(self.initiator_team) if self.initiator_team is not None else None,
                    "receiver_id": int(closest_id) if closest_id is not None else None,
                    "receiver_team": int(closest_team) if closest_team is not None else None,
                    "start_pos": [float(x) for x in self.pass_start_pos] if self.pass_start_pos is not None else None,
                    "end_pos": [float(x) for x in ball_pos] if ball_pos is not None else None,
                    "distance_meters": round(float(distance_meters), 2),
                    "duration_seconds": round(float(duration_seconds), 2),
                    "start_frame": int(self.pass_start_frame) if self.pass_start_frame is not None else None,
                    "end_frame": int(frame_idx)
                }

                self.state = "POSSESSION"
                self.current_possessor = closest_id
                self.current_team = closest_team
                self.possession_frames = 1
                
                return event

            # FIX 2: Player passes to himself / Knock-on dribble (Silently reset)
            elif best_dist <= self.possession_radius and closest_id == self.initiator_id and ball_speed is not None and ball_speed < self.kick_speed_threshold:
                self.state = "POSSESSION"
                self.current_possessor = closest_id
                self.current_team = closest_team
                self.possession_frames = 1
                return None

        return None

