import numpy as np
import cv2
from collections import deque

class PossessionTracker:
    """
    Tracks ball possession frame-by-frame using 'Sticky' memory,
    retaining possession during passes/dribbles until the opponent wins it.
    """
    def __init__(self, possession_radius: float = 3.0, smoothing_window: int = 2):
        self.possession_radius = possession_radius
        self.smoothing_window = smoothing_window
        self._ignored_ids: set = {100, 469}
        self._raw_history: list[int] = []
        self._smoothed_history: list[int] = []
        self._window: deque[int] = deque(maxlen=smoothing_window)
        self.possession_frames: dict[int, int] = {1: 0, 2: 0, 0: 0}
        
        # STICKY MEMORY
        self._last_team: int = 0
        self.last_player_id = None

    def update(self, ball_pos, player_tracks_frame: dict, goalkeeper_tracks_frame: dict | None = None) -> int:
        raw_team, raw_player = self._closest_player_and_team(ball_pos, player_tracks_frame)
        self._raw_history.append(raw_team)

        # STICKY LOGIC: Only update window and player if someone actively has the ball
        if raw_team != 0:
            self._window.append(raw_team)
            self.last_player_id = raw_player

        smoothed = self._majority_vote()
        self._smoothed_history.append(smoothed)
        self.possession_frames[smoothed] = self.possession_frames.get(smoothed, 0) + 1
        self._last_team = smoothed
        return smoothed

    def get_possession_percentages(self) -> tuple[float, float]:
        t1 = self.possession_frames.get(1, 0)
        t2 = self.possession_frames.get(2, 0)
        total = t1 + t2
        if total == 0:
            return 50.0, 50.0
        return round(100.0 * t1 / total, 1), round(100.0 * t2 / total, 1)

    @property
    def current_team(self) -> int:
        return self._last_team

    @property
    def current_player(self):
        return self.last_player_id

    # --- DRAWING METHODS ---
    def get_possession_percentages_at(self, frame_idx: int) -> tuple[float, float]:
        history = self._smoothed_history[: frame_idx + 1]
        t1 = history.count(1)
        t2 = history.count(2)
        total = t1 + t2
        if total == 0:
            return 50.0, 50.0
        return round(100.0 * t1 / total, 1), round(100.0 * t2 / total, 1)

    def draw_possession_bar_at(self, frame: np.ndarray, frame_idx: int, team_colors: dict | None = None, position: str = "bottom") -> np.ndarray:
        current = self._smoothed_history[frame_idx] if frame_idx < len(self._smoothed_history) else 0
        t1_pct, t2_pct = self.get_possession_percentages_at(frame_idx)
        return self._render_bar(frame, t1_pct, t2_pct, current, team_colors, position)

    def draw_possession_bar(self, frame: np.ndarray, team_colors: dict | None = None, position: str = "bottom") -> np.ndarray:
        t1_pct, t2_pct = self.get_possession_percentages()
        return self._render_bar(frame, t1_pct, t2_pct, self._last_team, team_colors, position)

    def _render_bar(self, frame: np.ndarray, t1_pct: float, t2_pct: float, current_team: int, team_colors: dict | None = None, position: str = "bottom") -> np.ndarray:
        if team_colors is None:
            team_colors = {1: (0, 0, 220), 2: (220, 0, 0)}
        h, w = frame.shape[:2]
        bar_h = 36
        bar_w = int(w * 0.46)
        pad_x = int(w * 0.02)
        y_top = h - bar_h - 12 if position == "bottom" else 12
        y_bot = y_top + bar_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (pad_x, y_top), (pad_x + bar_w, y_bot), (20, 20, 20), -1)
        t1_w = int(bar_w * t1_pct / 100)
        cv2.rectangle(overlay, (pad_x, y_top), (pad_x + t1_w, y_bot), team_colors[1], -1)
        cv2.rectangle(overlay, (pad_x + t1_w, y_top), (pad_x + bar_w, y_bot), team_colors[2], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (pad_x, y_top), (pad_x + bar_w, y_bot), (255, 255, 255), 2)
        cv2.line(frame, (pad_x + t1_w, y_top), (pad_x + t1_w, y_bot), (255, 255, 255), 2)

        lbl = "POSSESSION"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        lx = pad_x + (bar_w - lw) // 2
        ly = y_top - 6
        cv2.putText(frame, lbl, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        fs = 0.62
        t1_txt = f"{t1_pct:.0f}%"
        (tw, _), _ = cv2.getTextSize(t1_txt, font, fs, 1)
        cv2.putText(frame, t1_txt, (pad_x + max(4, t1_w // 2 - tw // 2), y_top + bar_h - 8), font, fs, (255, 255, 255), 2)

        t2_txt = f"{t2_pct:.0f}%"
        (tw2, _), _ = cv2.getTextSize(t2_txt, font, fs, 1)
        t2_x = pad_x + t1_w + max(4, (bar_w - t1_w) // 2 - tw2 // 2)
        cv2.putText(frame, t2_txt, (t2_x, y_top + bar_h - 8), font, fs, (255, 255, 255), 2)

        dot_x = pad_x + t1_w // 2 if current_team == 1 else pad_x + t1_w + (bar_w - t1_w) // 2
        if current_team in (1, 2):
            cv2.circle(frame, (dot_x, y_top - 4), 5, (255, 255, 0), -1)

        return frame

    # --- PRIVATE HELPERS ---
    def _closest_player_and_team(self, ball_pos, player_tracks_frame: dict) -> tuple[int, int | None]:
        if ball_pos is None:
            return 0, None
        ball = np.array(ball_pos, dtype=np.float32)
        best_dist = np.inf
        best_team = 0
        best_player = None

        for tid, data in (player_tracks_frame or {}).items():
            if tid in self._ignored_ids:
                continue
            fp = data.get("field_pos")
            team = data.get("team")
            if fp is None or team is None:
                continue

            dist = float(np.linalg.norm(np.array(fp, dtype=np.float32) - ball))
            if dist < best_dist:
                best_dist = dist
                best_team = team
                best_player = tid

        if best_dist > self.possession_radius:
            return 0, None  
        return best_team, best_player

    def _majority_vote(self) -> int:
        if not self._window:
            return self._last_team
        counts = {1: 0, 2: 0}
        for v in self._window:
            counts[v] = counts.get(v, 0) + 1
        majority = max(counts, key=counts.__getitem__)
        if counts[majority] >= len(self._window) // 2 + 1:
            return majority
        return self._last_team