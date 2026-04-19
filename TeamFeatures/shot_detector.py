import numpy as np

class ShotDetector:
    def __init__(self, kick_speed_threshold=15.0):
        # Speed required to register as a shot (km/h)
        self.kick_speed_threshold = kick_speed_threshold
        
        # State tracking
        self.in_shot_motion = False
        self.shooter_id = None
        self.shooter_team = None
        self.start_pos = None
        self.start_frame = None

    def _is_in_attacking_third(self, x_pos, team_id):
        # Assuming Team 1 attacks Left->Right, Team 2 attacks Right->Left
        if team_id == 1 and x_pos > 70.0:  
            return True
        elif team_id == 2 and x_pos < 35.0:  
            return True
        return False

    def _is_6_yard_box(self, x, y):
        # The 6-yard box is 5.5m deep and 18.32m wide, centered at Y=34
        # Y bounds: 34 - 9.16 = 24.84 | 34 + 9.16 = 43.16
        in_y_range = 24.84 <= y <= 43.16
        in_left_box = (x <= 5.5) and in_y_range
        in_right_box = (x >= 99.5) and in_y_range
        return in_left_box or in_right_box

    def _is_penalty_box(self, x, y):
        # The 18-yard penalty box is 16.5m deep and 40.32m wide, centered at Y=34
        # Y bounds: 34 - 20.16 = 13.84 | 34 + 20.16 = 54.16
        in_y_range = 13.84 <= y <= 54.16
        in_left_box = (x <= 16.5) and in_y_range
        in_right_box = (x >= 88.5) and in_y_range
        return in_left_box or in_right_box

    def update(self, ball_pos, ball_speed, current_player, current_team, frame_idx):
        event = None

        # 1. DETECT SHOT INITIATION
        if not self.in_shot_motion and ball_speed > self.kick_speed_threshold:
            if current_player is not None and current_team in (1, 2):
                # Check if the player is close enough to the goal to shoot
                if self._is_in_attacking_third(ball_pos[0], current_team):
                    self.in_shot_motion = True
                    self.shooter_id = current_player
                    self.shooter_team = current_team
                    self.start_pos = ball_pos
                    self.start_frame = frame_idx

        # 2. TRACK SHOT OUTCOME
        elif self.in_shot_motion:
            # If the ball stops moving fast or changes possession, the shot is over
            if ball_speed < 5.0 or current_player != self.shooter_id:
                end_x, end_y = ball_pos[0], ball_pos[1]
                
                # --- THE GEOMETRIC FILTER ---
                if self._is_6_yard_box(end_x, end_y):
                    outcome = "ON_TARGET"
                elif self._is_penalty_box(end_x, end_y):
                    outcome = "OFF_TARGET / BLOCKED"
                else:
                    # The ball ended up completely outside the danger zones (e.g. out for a throw, or a wing pass)
                    # Therefore, this wasn't a shot at all. Cancel the event!
                    self.in_shot_motion = False
                    self.shooter_id = None
                    return None
                
                # Convert everything to standard Python ints and floats for JSON safety
                event = {
                    "event_type": "SHOT",
                    "shooter_id": int(self.shooter_id) if self.shooter_id is not None else None,
                    "shooter_team": int(self.shooter_team) if self.shooter_team is not None else None,
                    "start_pos": [float(x) for x in self.start_pos],
                    "end_pos": [float(x) for x in ball_pos],
                    "outcome": outcome,
                    "start_frame": int(self.start_frame),
                    "end_frame": int(frame_idx)
                }
                
                # Reset state
                self.in_shot_motion = False
                self.shooter_id = None
                
        return event