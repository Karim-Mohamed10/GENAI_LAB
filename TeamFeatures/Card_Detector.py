from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


class CardDetector:
	def __init__(
		self,
		model_path: str = "models/cards.pt",
		conf: float = 0.25,
		iou: float = 0.5,
		event_cooldown_frames: int = 45,
		association_expand_ratio: float = 0.35,
	):
		self.model = YOLO(model_path)
		self.conf = conf
		self.iou = iou
		self.event_cooldown_frames = event_cooldown_frames
		self.association_expand_ratio = association_expand_ratio

		# Per-player caps requested by user.
		self.max_cards_per_player = {"red": 1, "yellow": 2}

		self.player_card_counts = defaultdict(lambda: {"red": 0, "yellow": 0, "team": None})
		self.team_card_totals = {
			1: {"red": 0, "yellow": 0},
			2: {"red": 0, "yellow": 0},
		}

		self._last_counted_event_frame = {}

	@staticmethod
	def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
		x1, y1, x2, y2 = bbox
		return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))

	@staticmethod
	def _normalize_class_name(name: str) -> Optional[str]:
		text = str(name).strip().lower()
		if "red" in text:
			return "red"
		if "yellow" in text:
			return "yellow"
		return None

	def _expand_bbox(self, bbox: List[float]) -> List[float]:
		x1, y1, x2, y2 = bbox
		w = x2 - x1
		h = y2 - y1
		pad_w = w * self.association_expand_ratio
		pad_h = h * self.association_expand_ratio
		return [x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h]

	@staticmethod
	def _point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
		px, py = point
		x1, y1, x2, y2 = bbox
		return x1 <= px <= x2 and y1 <= py <= y2

	def detect_cards(self, frame: np.ndarray) -> List[Dict]:
		result = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
		detections = []

		names = result.names
		for box in result.boxes:
			cls_id = int(box.cls[0].item())
			class_name = names.get(cls_id, str(cls_id))
			card_type = self._normalize_class_name(class_name)
			if card_type is None:
				continue

			xyxy = box.xyxy[0].tolist()
			conf = float(box.conf[0].item())
			detections.append(
				{
					"bbox": xyxy,
					"conf": conf,
					"card_type": card_type,
					"class_name": class_name,
				}
			)

		return detections

	def _associate_card_to_player(self, card_bbox: List[float], players_in_frame: Dict) -> Optional[int]:
		if not players_in_frame:
			return None

		card_center = self._bbox_center(card_bbox)
		containing_candidates = []
		nearest_candidate = None
		nearest_dist = float("inf")

		for player_id, pdata in players_in_frame.items():
			player_bbox = pdata.get("bbox")
			if player_bbox is None:
				continue

			player_center = self._bbox_center(player_bbox)
			dist = float(np.linalg.norm(np.array(card_center) - np.array(player_center)))

			if dist < nearest_dist:
				nearest_dist = dist
				nearest_candidate = player_id

			expanded_bbox = self._expand_bbox(player_bbox)
			if self._point_in_bbox(card_center, expanded_bbox):
				containing_candidates.append((player_id, dist))

		if containing_candidates:
			containing_candidates.sort(key=lambda item: item[1])
			return containing_candidates[0][0]

		return nearest_candidate

	def update(self, frame: np.ndarray, players_in_frame: Dict, frame_index: int) -> List[Dict]:
		detections = self.detect_cards(frame)
		counted_events = []

		for det in detections:
			card_type = det["card_type"]
			player_id = self._associate_card_to_player(det["bbox"], players_in_frame)
			if player_id is None:
				continue

			player_stats = self.player_card_counts[player_id]
			cap = self.max_cards_per_player[card_type]
			if player_stats[card_type] >= cap:
				continue

			event_key = (player_id, card_type)
			last_counted = self._last_counted_event_frame.get(event_key, -10**9)
			if frame_index - last_counted < self.event_cooldown_frames:
				continue

			player_stats[card_type] += 1
			self._last_counted_event_frame[event_key] = frame_index

			team_id = players_in_frame.get(player_id, {}).get("team")
			if team_id in (1, 2):
				player_stats["team"] = team_id
				self.team_card_totals[team_id][card_type] += 1

			counted_events.append(
				{
					"frame": frame_index,
					"player_id": player_id,
					"team": player_stats.get("team"),
					"card_type": card_type,
					"conf": det["conf"],
				}
			)

		return counted_events

	def get_summary(self) -> Dict:
		per_player = {}
		for player_id, stats in self.player_card_counts.items():
			if stats["red"] == 0 and stats["yellow"] == 0:
				continue
			per_player[int(player_id)] = {
				"team": stats.get("team"),
				"red": int(stats["red"]),
				"yellow": int(stats["yellow"]),
			}

		return {
			"team_totals": self.team_card_totals,
			"per_player": dict(sorted(per_player.items(), key=lambda kv: kv[0])),
		}
