from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _PendingTackle:
    tackler_id: int
    tackler_team: int
    victim_id: int
    victim_team: int
    start_frame: int
    last_contact_frame: int


class TackleDetector:
    """
    Rule-based tackle detector that works on tracked players and possession stream.

    A tackle attempt is started when an opponent gets close to the current
    ball carrier. The attempt is finalized as:
    - SUCCESS: possession changes to tackler's team soon after contact.
    - FAILED: possession does not change within the allowed outcome window.

    This class is intentionally standalone and does not depend on the main
    pipeline integration. You can feed one frame at a time from any loop.
    """

    def __init__(
        self,
        contact_distance_m: float = 2.0,
        release_distance_m: float = 2.8,
        min_contact_frames: int = 3,
        outcome_window_frames: int = 20,
        cooldown_frames: int = 12,
        hard_close_distance_m: float = 1.25,
        min_relative_speed_mps: float = 0.9,
        foul_image_model_path: str | None = "models/tackle_foul_classifier.keras",
        foul_event_model_path: str | None = "models/tackle_event_classifier.joblib",
        foul_config_path: str | None = "models/tackle_foul_config.json",
        fps: float = 25.0,
    ) -> None:
        self.contact_distance_m = contact_distance_m
        self.release_distance_m = release_distance_m
        self.min_contact_frames = min_contact_frames
        self.outcome_window_frames = outcome_window_frames
        self.cooldown_frames = cooldown_frames
        self.hard_close_distance_m = hard_close_distance_m
        self.min_relative_speed_mps = min_relative_speed_mps
        self.fps = fps

        self._pending: _PendingTackle | None = None
        self._contact_frames: int = 0
        self._last_event_frame: int = -10**9

        self._events: list[dict[str, Any]] = []
        self._player_stats: dict[int, dict[str, int]] = {}
        self._team_stats: dict[int, dict[str, int]] = {1: self._empty_stats(), 2: self._empty_stats()}

        self._foul_image_model = None
        self._foul_event_model = None
        self._foul_decision_threshold = 0.5
        self._load_models(foul_image_model_path, foul_event_model_path)
        self._load_foul_config(foul_config_path)

    def _load_models(self, foul_image_model_path: str | None, foul_event_model_path: str | None) -> None:
        if foul_image_model_path:
            try:
                model_path = Path(foul_image_model_path)
                if model_path.exists():
                    import tensorflow as tf

                    self._foul_image_model = tf.keras.models.load_model(str(model_path))
                    print(f"Loaded foul image model: {model_path}")
            except Exception as exc:
                print(f"WARNING: Could not load foul image model: {exc}")

        if foul_event_model_path:
            try:
                model_path = Path(foul_event_model_path)
                if model_path.exists():
                    import joblib

                    self._foul_event_model = joblib.load(str(model_path))
                    print(f"Loaded foul event model: {model_path}")
            except Exception as exc:
                print(f"WARNING: Could not load foul event model: {exc}")

    def _load_foul_config(self, foul_config_path: str | None) -> None:
        if not foul_config_path:
            return

        try:
            cfg_path = Path(foul_config_path)
            if not cfg_path.exists():
                return

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            threshold = cfg.get("foul_decision_threshold")
            if threshold is not None:
                self._foul_decision_threshold = float(threshold)
                print(f"Loaded foul threshold: {self._foul_decision_threshold:.3f} from {cfg_path}")
        except Exception as exc:
            print(f"WARNING: Could not load foul config: {exc}")

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {"tackles": 0, "successes": 0, "fails": 0}

    def update(
        self,
        frame_idx: int,
        players_frame: dict[int, dict[str, Any]],
        possessing_player_id: int | None,
        possessing_team_id: int | None,
        frame_image: Any | None = None,
        ball_position: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process one frame and return a list of finalized tackle events.

        Expected player structure per frame:
        {
            player_id: {
                "field_pos": (x, y),
                "team": 1 or 2,
                ...
            }
        }
        """
        new_events: list[dict[str, Any]] = []

        # If possession is unknown, try to close a pending tackle by timeout.
        if possessing_player_id is None or possessing_team_id not in (1, 2):
            timeout_event = self._try_finalize_by_timeout(frame_idx, possession_changed=False)
            if timeout_event is not None:
                new_events.append(timeout_event)
            return new_events

        possessor = players_frame.get(possessing_player_id)
        possessor_pos = self._get_pos(possessor)
        if possessor is None or possessor_pos is None:
            timeout_event = self._try_finalize_by_timeout(frame_idx, possession_changed=False)
            if timeout_event is not None:
                new_events.append(timeout_event)
            return new_events

        # Existing pending tackle can be resolved by possession change.
        if self._pending is not None:
            # Verify the victim still has possession, or resolve based on possession change
            if possessing_player_id != self._pending.victim_id:
                # Possession changed to someone else - check if it's to the tackler (success)
                if possessing_team_id == self._pending.tackler_team:
                    event = self._finalize_pending(frame_idx=frame_idx, success=True, frame_image=frame_image, players_frame=players_frame)
                    if event is not None:
                        new_events.append(event)
                    return new_events
                else:
                    # Possession changed but not to tackler - fail the tackle
                    event = self._finalize_pending(frame_idx=frame_idx, success=False, frame_image=frame_image, players_frame=players_frame)
                    if event is not None:
                        new_events.append(event)
                    return new_events
            
            # Victim still has possession - track contact continuity
            if possessing_team_id == self._pending.tackler_team:
                event = self._finalize_pending(frame_idx=frame_idx, success=True, frame_image=frame_image, players_frame=players_frame)
                if event is not None:
                    new_events.append(event)
                return new_events

            current_tackler = players_frame.get(self._pending.tackler_id)
            current_tackler_pos = self._get_pos(current_tackler)
            victim_now = players_frame.get(self._pending.victim_id)
            victim_now_pos = self._get_pos(victim_now)
            reference_pos = victim_now_pos if victim_now_pos is not None else possessor_pos
            if current_tackler_pos is not None:
                d = self._distance(current_tackler_pos, reference_pos)
                if d <= self.contact_distance_m:
                    self._contact_frames += 1
                    self._pending.last_contact_frame = frame_idx
                elif d > self.release_distance_m:
                    # Contact ended; wait only outcome window for possession switch.
                    pass

            timeout_event = self._try_finalize_by_timeout(frame_idx, possession_changed=False)
            if timeout_event is not None:
                new_events.append(timeout_event)

            return new_events

        # Cooldown to avoid repeated duplicate attempts in a short burst.
        if frame_idx - self._last_event_frame < self.cooldown_frames:
            return new_events

        # No pending tackle: search for nearest opponent to current possessor.
        nearest_opp_id, nearest_opp_team, nearest_dist, nearest_opp_data = self._find_nearest_opponent(
            possessor_id=possessing_player_id,
            possessor_team=possessing_team_id,
            possessor_pos=possessor_pos,
            players_frame=players_frame,
        )

        if nearest_opp_id is None or nearest_opp_team is None:
            return new_events

        victim_id, victim_team, victim_data = self._find_victim_for_tackler(
            tackler_id=nearest_opp_id,
            tackler_team=nearest_opp_team,
            players_frame=players_frame,
            ball_position=ball_position,
            fallback_victim_id=possessing_player_id,
            fallback_victim_team=possessing_team_id,
        )

        if victim_id is None or victim_team is None:
            return new_events

        if nearest_dist <= self.contact_distance_m and self._is_intense_challenge(
            tackler_data=nearest_opp_data,
            victim_data=victim_data,
            distance_m=nearest_dist,
        ):
            self._pending = _PendingTackle(
                tackler_id=nearest_opp_id,
                tackler_team=nearest_opp_team,
                victim_id=victim_id,
                victim_team=victim_team,
                start_frame=frame_idx,
                last_contact_frame=frame_idx,
            )
            self._contact_frames = 1

        return new_events

    def _try_finalize_by_timeout(self, frame_idx: int, possession_changed: bool) -> dict[str, Any] | None:
        if self._pending is None:
            return None

        no_contact_for = frame_idx - self._pending.last_contact_frame
        if no_contact_for < self.outcome_window_frames:
            return None

        return self._finalize_pending(frame_idx=frame_idx, success=possession_changed)

    def _finalize_pending(
        self,
        frame_idx: int,
        success: bool,
        frame_image: Any | None = None,
        players_frame: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if self._pending is None:
            return None

        # Ignore extremely short contacts as noise.
        if self._contact_frames < self.min_contact_frames:
            self._pending = None
            self._contact_frames = 0
            return None

        outcome = "success" if success else "failed"
        event = {
            "event_type": "TACKLE",
            "status": outcome.upper(),
            "outcome": outcome,
            "tackler_id": int(self._pending.tackler_id),
            "tackler_team": int(self._pending.tackler_team),
            "victim_id": int(self._pending.victim_id),
            "victim_team": int(self._pending.victim_team),
            "start_frame": int(self._pending.start_frame),
            "end_frame": int(frame_idx),
            "contact_frames": int(self._contact_frames),
        }

        self._attach_model_predictions(
            event=event,
            frame_idx=frame_idx,
            frame_image=frame_image,
            players_frame=players_frame,
        )

        self._events.append(event)
        self._record_stats(
            tackler_id=self._pending.tackler_id,
            tackler_team=self._pending.tackler_team,
            success=success,
        )

        self._last_event_frame = frame_idx
        self._pending = None
        self._contact_frames = 0
        return event

    def _attach_model_predictions(
        self,
        event: dict[str, Any],
        frame_idx: int,
        frame_image: Any | None,
        players_frame: dict[int, dict[str, Any]] | None,
    ) -> None:
        img_prob = self._predict_foul_prob_from_image(event, frame_image, players_frame)
        evt_prob = self._predict_foul_prob_from_event(event, frame_idx, players_frame)

        probs = [p for p in [img_prob, evt_prob] if p is not None]
        if probs:
            final_prob = float(sum(probs) / len(probs))
            event["foul_probability"] = round(final_prob, 4)
            event["foul_threshold"] = round(float(self._foul_decision_threshold), 4)
            event["is_foul_model"] = bool(final_prob >= self._foul_decision_threshold)

        if img_prob is not None:
            event["foul_probability_image"] = round(float(img_prob), 4)
        if evt_prob is not None:
            event["foul_probability_event"] = round(float(evt_prob), 4)

    def _predict_foul_prob_from_image(
        self,
        event: dict[str, Any],
        frame_image: Any | None,
        players_frame: dict[int, dict[str, Any]] | None,
    ) -> float | None:
        if self._foul_image_model is None or frame_image is None or players_frame is None:
            return None

        tackler = players_frame.get(event.get("tackler_id"))
        if not tackler:
            return None

        bbox = tackler.get("bbox")
        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            import cv2

            crop = cv2.resize(crop, (224, 224))
            crop = crop.astype(np.float32) / 255.0
            batch = np.expand_dims(crop, axis=0)
            pred = self._foul_image_model.predict(batch, verbose=0)
            return float(np.squeeze(pred))
        except Exception:
            return None

    def _predict_foul_prob_from_event(
        self,
        event: dict[str, Any],
        frame_idx: int,
        players_frame: dict[int, dict[str, Any]] | None,
    ) -> float | None:
        if self._foul_event_model is None or players_frame is None:
            return None

        victim = players_frame.get(event.get("victim_id"), {})
        pos = victim.get("field_pos")
        if pos is None or len(pos) != 2:
            return None

        minute = (frame_idx / max(self.fps, 1.0)) / 60.0
        second = (frame_idx / max(self.fps, 1.0)) % 60.0
        duration = event.get("contact_frames", 0) / max(self.fps, 1.0)

        features = np.array(
            [[float(pos[0]), float(pos[1]), float(minute), float(second), 1.0, float(duration)]],
            dtype=np.float32,
        )

        try:
            if hasattr(self._foul_event_model, "predict_proba"):
                proba = self._foul_event_model.predict_proba(features)
                if proba.shape[1] >= 2:
                    return float(proba[0][1])
            pred = self._foul_event_model.predict(features)
            return float(pred[0])
        except Exception:
            return None

    def _record_stats(self, tackler_id: int, tackler_team: int, success: bool) -> None:
        if tackler_id not in self._player_stats:
            self._player_stats[tackler_id] = self._empty_stats()

        if tackler_team not in self._team_stats:
            self._team_stats[tackler_team] = self._empty_stats()

        self._player_stats[tackler_id]["tackles"] += 1
        self._team_stats[tackler_team]["tackles"] += 1

        if success:
            self._player_stats[tackler_id]["successes"] += 1
            self._team_stats[tackler_team]["successes"] += 1
        else:
            self._player_stats[tackler_id]["fails"] += 1
            self._team_stats[tackler_team]["fails"] += 1

    @staticmethod
    def _get_pos(player_data: dict[str, Any] | None) -> tuple[float, float] | None:
        if not player_data:
            return None

        pos = player_data.get("field_pos")
        if pos is None or len(pos) != 2:
            return None

        return float(pos[0]), float(pos[1])

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _get_speed_mps(player_data: dict[str, Any] | None) -> float:
        if not player_data:
            return 0.0

        speed_kmh = player_data.get("speed")
        if speed_kmh is None:
            return 0.0

        try:
            return max(0.0, float(speed_kmh) / 3.6)
        except Exception:
            return 0.0

    def _is_intense_challenge(
        self,
        tackler_data: dict[str, Any] | None,
        victim_data: dict[str, Any] | None,
        distance_m: float,
    ) -> bool:
        # Always allow very close body contact.
        if distance_m <= self.hard_close_distance_m:
            return True

        # Otherwise require a meaningful speed differential to avoid walk-by false positives.
        tackler_speed = self._get_speed_mps(tackler_data)
        victim_speed = self._get_speed_mps(victim_data)
        return (tackler_speed - victim_speed) >= self.min_relative_speed_mps

    def _find_nearest_opponent(
        self,
        possessor_id: int,
        possessor_team: int,
        possessor_pos: tuple[float, float],
        players_frame: dict[int, dict[str, Any]],
    ) -> tuple[int | None, int | None, float, dict[str, Any] | None]:
        best_id: int | None = None
        best_team: int | None = None
        best_dist: float = float("inf")
        best_data: dict[str, Any] | None = None

        for pid, pdata in players_frame.items():
            if pid == possessor_id:
                continue

            team = pdata.get("team")
            if team not in (1, 2) or team == possessor_team:
                continue

            pos = self._get_pos(pdata)
            if pos is None:
                continue

            d = self._distance(possessor_pos, pos)
            if d < best_dist:
                best_dist = d
                best_id = int(pid)
                best_team = int(team)
                best_data = pdata

        return best_id, best_team, best_dist, best_data

    def _find_victim_for_tackler(
        self,
        tackler_id: int,
        tackler_team: int,
        players_frame: dict[int, dict[str, Any]],
        ball_position: tuple[float, float] | None,
        fallback_victim_id: int | None,
        fallback_victim_team: int | None,
    ) -> tuple[int | None, int | None, dict[str, Any] | None]:
        tackler = players_frame.get(tackler_id)
        tackler_pos = self._get_pos(tackler)
        if tackler_pos is None:
            if fallback_victim_id is None or fallback_victim_team is None:
                return None, None, None
            return fallback_victim_id, fallback_victim_team, players_frame.get(fallback_victim_id)

        ball_point = None
        if ball_position is not None and len(ball_position) == 2:
            ball_point = (float(ball_position[0]), float(ball_position[1]))

        best_id: int | None = None
        best_team: int | None = None
        best_data: dict[str, Any] | None = None
        best_score = float("inf")

        for pid, pdata in players_frame.items():
            if pid == tackler_id:
                continue

            team = pdata.get("team")
            if team not in (1, 2) or team == tackler_team:
                continue

            pos = self._get_pos(pdata)
            if pos is None:
                continue

            d_tackler = self._distance(tackler_pos, pos)
            d_ball = self._distance(pos, ball_point) if ball_point is not None else 0.0
            # Prioritize opponent closest to tackler and also close to ball.
            score = d_tackler + (0.7 * d_ball)

            if score < best_score:
                best_score = score
                best_id = int(pid)
                best_team = int(team)
                best_data = pdata

        if best_id is not None and best_team is not None:
            return best_id, best_team, best_data

        if fallback_victim_id is None or fallback_victim_team is None:
            return None, None, None
        return fallback_victim_id, fallback_victim_team, players_frame.get(fallback_victim_id)

    def get_player_stats(self) -> dict[int, dict[str, int]]:
        """Return per-player tackle totals and outcomes."""
        return {pid: stats.copy() for pid, stats in self._player_stats.items()}

    def get_team_stats(self) -> dict[int, dict[str, int]]:
        """Return per-team tackle totals and outcomes."""
        return {tid: stats.copy() for tid, stats in self._team_stats.items()}

    def get_events(self) -> list[dict[str, Any]]:
        """Return all finalized tackle events in chronological order."""
        return [evt.copy() for evt in self._events]

    def reset(self) -> None:
        self._pending = None
        self._contact_frames = 0
        self._last_event_frame = -10**9
        self._events = []
        self._player_stats = {}
        self._team_stats = {1: self._empty_stats(), 2: self._empty_stats()}
