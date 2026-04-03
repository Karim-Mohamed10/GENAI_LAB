import math

class PassDetector:
    def __init__(self, possession_radius=2.0, kick_speed_threshold=15.0, min_possession_frames=3):
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
            if ball_speed is not None and ball_speed < self.kick_speed_threshold and best_dist <= self.possession_radius:
                status = "COMPLETED" if closest_team == self.initiator_team else "INTERCEPTED"
                
                distance_meters = 0.0
                if self.pass_start_pos and ball_pos:
                    distance_meters = math.hypot(ball_pos[0] - self.pass_start_pos[0], ball_pos[1] - self.pass_start_pos[1])
                
                duration_seconds = 0.0
                if self.pass_start_frame is not None and fps:
                    duration_seconds = (frame_idx - self.pass_start_frame) / fps

                event = {
                    "event_type": "PASS",
                    "status": status,
                    "initiator_id": self.initiator_id,
                    "initiator_team": self.initiator_team,
                    "receiver_id": closest_id,
                    "receiver_team": closest_team,
                    "start_pos": self.pass_start_pos,
                    "end_pos": ball_pos,
                    "distance_meters": distance_meters,
                    "duration_seconds": duration_seconds
                }

                self.state = "POSSESSION"
                self.current_possessor = closest_id
                self.current_team = closest_team
                self.possession_frames = 1
                
                return event

        return None
