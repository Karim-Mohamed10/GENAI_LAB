import numpy as np
import cv2
from collections import deque


class PossessionTracker:
    """
    Tracks ball possession frame-by-frame based on the closest player
    to the ball in real-world field coordinates.

    Parameters
    ----------
    possession_radius : float
        Maximum distance (metres) between ball and player centre for
        that player to be considered 'in possession'.  If no player is
        within this radius the frame is marked as contested (team 0).
    smoothing_window : int
        Rolling window size used to smooth raw per-frame possession
        before deciding the displayed team.  Higher values = more
        stable but slower to react.
    """

    def __init__(self, possession_radius: float = 3.0, smoothing_window: int = 5):
        self.possession_radius = possession_radius
        self.smoothing_window = smoothing_window

        # Track IDs permanently excluded from possession (e.g. goalkeepers)
        self._ignored_ids: set = {100, 469}

        # Raw per-frame possession: 1, 2, or 0 (contested)
        self._raw_history: list[int] = []
        # Smoothed per-frame possession stored for stats
        self._smoothed_history: list[int] = []

        # Sliding window for smoothing
        self._window: deque[int] = deque(maxlen=smoothing_window)

        # Cumulative frame counts per team (smoothed)
        self.possession_frames: dict[int, int] = {1: 0, 2: 0, 0: 0}

        # Last decided team (for display continuity)
        self._last_team: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        ball_pos,
        player_tracks_frame: dict,
        goalkeeper_tracks_frame: dict | None = None,
    ) -> int:
        """
        Determine which team has possession in the current frame.
        Goalkeepers are intentionally excluded from possession calculation.

        Parameters
        ----------
        ball_pos : array-like of shape (2,) or None
            Ball position in field coordinates [x, y] (metres).
        player_tracks_frame : dict
            ``{track_id: {"field_pos": (x, y), "team": int, ...}}``
            for outfield players in this frame.
        goalkeeper_tracks_frame : dict, optional
            Ignored. Goalkeepers are excluded from possession calculation.

        Returns
        -------
        int
            Smoothed team id with possession: 1, 2, or 0 (contested).
        """
        raw_team = self._closest_team(ball_pos, player_tracks_frame)
        self._raw_history.append(raw_team)

        # Push to smoothing window (skip contested frames so short
        # gaps don't reset possession unnecessarily)
        if raw_team != 0:
            self._window.append(raw_team)

        # Decide smoothed possession: majority vote in window
        smoothed = self._majority_vote()
        self._smoothed_history.append(smoothed)
        self.possession_frames[smoothed] = self.possession_frames.get(smoothed, 0) + 1
        self._last_team = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def get_possession_percentages(self) -> tuple[float, float]:
        """
        Return (team1_pct, team2_pct) rounded to one decimal place.
        Contested frames are excluded from the denominator so that the
        two values always sum to 100 %.
        """
        t1 = self.possession_frames.get(1, 0)
        t2 = self.possession_frames.get(2, 0)
        total = t1 + t2
        if total == 0:
            return 50.0, 50.0
        return round(100.0 * t1 / total, 1), round(100.0 * t2 / total, 1)

    @property
    def current_team(self) -> int:
        """Team id that currently has possession (smoothed)."""
        return self._last_team

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def get_possession_percentages_at(self, frame_idx: int) -> tuple[float, float]:
        """
        Return (team1_pct, team2_pct) computed only up to *frame_idx*
        (inclusive).  Useful for per-frame replay.
        """
        history = self._smoothed_history[: frame_idx + 1]
        t1 = history.count(1)
        t2 = history.count(2)
        total = t1 + t2
        if total == 0:
            return 50.0, 50.0
        return round(100.0 * t1 / total, 1), round(100.0 * t2 / total, 1)

    def draw_possession_bar_at(
        self,
        frame: np.ndarray,
        frame_idx: int,
        team_colors: dict | None = None,
        position: str = "bottom",
    ) -> np.ndarray:
        """
        Like :meth:`draw_possession_bar` but shows stats only up to
        *frame_idx* so the bar updates correctly during video replay.
        """
        current = self._smoothed_history[frame_idx] if frame_idx < len(self._smoothed_history) else 0
        t1_pct, t2_pct = self.get_possession_percentages_at(frame_idx)
        return self._render_bar(frame, t1_pct, t2_pct, current, team_colors, position)

    def draw_possession_bar(
        self,
        frame: np.ndarray,
        team_colors: dict | None = None,
        position: str = "bottom",
    ) -> np.ndarray:
        """
        Overlay a possession bar on *frame* and return the result.

        Parameters
        ----------
        frame : np.ndarray
            BGR video frame to draw on (modified in-place).
        team_colors : dict, optional
            ``{1: (B, G, R), 2: (B, G, R)}`` override colours.
            Defaults to red vs blue.
        position : {"bottom", "top"}
            Where on the frame to place the bar.
        """
        t1_pct, t2_pct = self.get_possession_percentages()
        return self._render_bar(frame, t1_pct, t2_pct, self._last_team, team_colors, position)

    def _render_bar(
        self,
        frame: np.ndarray,
        t1_pct: float,
        t2_pct: float,
        current_team: int,
        team_colors: dict | None = None,
        position: str = "bottom",
    ) -> np.ndarray:
        # Default BGR colours: team1 = red, team2 = blue
        if team_colors is None:
            team_colors = {1: (0, 0, 220), 2: (220, 0, 0)}

        h, w = frame.shape[:2]
        bar_h = 36
        bar_w = int(w * 0.46)
        pad_x = int(w * 0.02)

        y_top = h - bar_h - 12 if position == "bottom" else 12
        y_bot = y_top + bar_h

        # --- background ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad_x, y_top), (pad_x + bar_w, y_bot), (20, 20, 20), -1)

        # --- team 1 fill ---
        t1_w = int(bar_w * t1_pct / 100)
        cv2.rectangle(overlay, (pad_x, y_top), (pad_x + t1_w, y_bot), team_colors[1], -1)

        # --- team 2 fill ---
        cv2.rectangle(
            overlay, (pad_x + t1_w, y_top), (pad_x + bar_w, y_bot), team_colors[2], -1
        )

        # Blend with 80 % opacity so the field is still visible
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # --- border ---
        cv2.rectangle(frame, (pad_x, y_top), (pad_x + bar_w, y_bot), (255, 255, 255), 2)
        # --- divider line ---
        cv2.line(frame, (pad_x + t1_w, y_top), (pad_x + t1_w, y_bot), (255, 255, 255), 2)

        # --- label: "POSSESSION" ---
        lbl = "POSSESSION"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        lx = pad_x + (bar_w - lw) // 2
        ly = y_top - 6
        cv2.putText(frame, lbl, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1)

        # --- percentage texts ---
        font = cv2.FONT_HERSHEY_DUPLEX
        fs = 0.62

        t1_txt = f"{t1_pct:.0f}%"
        (tw, _), _ = cv2.getTextSize(t1_txt, font, fs, 1)
        cv2.putText(
            frame, t1_txt,
            (pad_x + max(4, t1_w // 2 - tw // 2), y_top + bar_h - 8),
            font, fs, (255, 255, 255), 2,
        )

        t2_txt = f"{t2_pct:.0f}%"
        (tw2, _), _ = cv2.getTextSize(t2_txt, font, fs, 1)
        t2_x = pad_x + t1_w + max(4, (bar_w - t1_w) // 2 - tw2 // 2)
        cv2.putText(
            frame, t2_txt, (t2_x, y_top + bar_h - 8),
            font, fs, (255, 255, 255), 2,
        )

        # --- possession indicator dot above current-possession side ---
        dot_x = pad_x + t1_w // 2 if current_team == 1 else pad_x + t1_w + (bar_w - t1_w) // 2
        if current_team in (1, 2):
            cv2.circle(frame, (dot_x, y_top - 4), 5, (255, 255, 0), -1)

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _closest_team(
        self,
        ball_pos,
        player_tracks_frame: dict,
    ) -> int:
        """Return team id of the outfield player closest to the ball, or 0."""
        if ball_pos is None:
            return 0

        ball = np.array(ball_pos, dtype=np.float32)

        best_dist = np.inf
        best_team = 0

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

        if best_dist > self.possession_radius:
            return 0  # contested – no one close enough

        return best_team

    def _majority_vote(self) -> int:
        """Return majority team in the current smoothing window."""
        if not self._window:
            return self._last_team

        counts = {1: 0, 2: 0}
        for v in self._window:
            counts[v] = counts.get(v, 0) + 1

        majority = max(counts, key=counts.__getitem__)
        # Only switch if majority wins at least half the window
        if counts[majority] >= len(self._window) // 2 + 1:
            return majority
        return self._last_team
