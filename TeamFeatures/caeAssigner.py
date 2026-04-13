from collections import defaultdict, deque
import importlib
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans

keras = None
CAE_MODEL_PATH = "models/cae_best.keras"


class CAETeamAssigner:
    def __init__(
        self,
        stability_window=5,
        input_size=64,
        latent_dim=128,
        reassign_interval=5,
    ):
  
        self.team_colors = {1: (0, 0, 255), 2: (255, 0, 0)}
        self.player_team_dict = {}
        self.kmeans = None
        self.stability_window = stability_window
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.reassign_interval = reassign_interval
        self.encoder = None
        self.assignment_mode = "color"

        self.team_feature_centers = {}
        self.cluster_to_team = {}
        self.team_cluster_ids = (0, 1)
        self._last_eval_frame = {}
        self._last_raw_team = {}

        # Per-player sliding window of raw per-frame votes
        self._vote_history = defaultdict(lambda: deque(maxlen=stability_window))

    def _display_colors(self):
        return {1: (0, 0, 255), 2: (255, 0, 0)}

    def _set_display_colors(self):
        self.team_colors = self._display_colors()

    def load_encoder(self):
        if self.encoder is not None:
            return True
        global keras
        if keras is None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            try:
                tf_module = importlib.import_module("tensorflow")
            except Exception:
                return False
            try:
                tf_module.get_logger().setLevel("ERROR")
            except Exception:
                pass
            keras = tf_module.keras

        if keras is None:
            return False

        model = keras.models.load_model(CAE_MODEL_PATH)

        output_shape = getattr(model, "output_shape", None)
        if output_shape is not None:
            try:
                is_latent_model = len(output_shape) == 2
            except TypeError:
                is_latent_model = False
        else:
            is_latent_model = False

        if not is_latent_model:
            model = keras.Model(
                inputs=model.input,
                outputs=model.get_layer("latent").output,
                name="CAE_Encoder",
            )

        self.encoder = model
        return True

    def get_player_crop(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, min(w_frame, x1))
        y1 = max(0, min(h_frame, y1))
        x2 = max(0, min(w_frame, x2))
        y2 = max(0, min(h_frame, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        crop = crop.astype(np.float32) / 255.0
        return crop

    def extract_player_features(self, frame, bbox):
        if not self.load_encoder():
            return None

        crop = self.get_player_crop(frame, bbox)
        if crop is None:
            return None

        features = self.encoder.predict(crop[np.newaxis, ...], verbose=0)
        features = np.asarray(features, dtype=np.float32).reshape(-1)
        return features

    def _update_and_get_stable_team(self, player_id, raw_team):
        if raw_team is not None:
            self._vote_history[player_id].append(raw_team)

        history = self._vote_history[player_id]

        if len(history) >= self.stability_window and len(set(history)) == 1:
            self.player_team_dict[player_id] = history[0]

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if history:
            return max(set(history), key=list(history).count)

        return 1

    def get_player_color(self, frame, bbox):
        """Extract dominant player color by masking out grass and normalizing brightness."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h_frame, w_frame = frame.shape[:2]

        # 1. ULTRA-TIGHT CROP (Avoid shorts, hair, and ARMS/SKIN)
        h_box, w_box = y2 - y1, x2 - x1
        crop_y1 = max(0, y1 + int(h_box * 0.15))
        crop_y2 = min(h_frame, y1 + int(h_box * 0.55))
        crop_x1 = max(0, x1 + int(w_box * 0.30))
        crop_x2 = min(w_frame, x2 - int(w_box * 0.30))

        image_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if image_crop.shape[0] < 2 or image_crop.shape[1] < 2:
            return None

        # 2. CONVERT TO HSV FOR MASKING
        hsv_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

        # 3. BACKGROUND MASKING (Remove Grass)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv_crop, lower_green, upper_green)
        mask_player = cv2.bitwise_not(mask_green)

        # 4. EXTRACT JERSEY PIXELS
        player_pixels = image_crop[mask_player == 255]
        if len(player_pixels) < 10:
            player_pixels = image_crop.reshape(-1, 3)

        # 5. NORMALIZE BRIGHTNESS
        pixels_reshaped = player_pixels.reshape(-1, 1, 3)
        hsv_pixels = cv2.cvtColor(pixels_reshaped, cv2.COLOR_BGR2HSV)
        hsv_pixels[:, :, 2] = 200
        normalized_bgr = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR).reshape(-1, 3)

        # 6. FIND TRUE DOMINANT COLOR
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=42)
        kmeans.fit(normalized_bgr)

        labels = kmeans.labels_
        counts = np.bincount(labels)
        dominant_cluster = np.argmax(counts)

        return kmeans.cluster_centers_[dominant_cluster]

    def assign_team_color(self, frame, player_detections):
        """
        Determine the two team colors by clustering the dominant colors
        of all detected players in a single frame.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            if player_color is not None:
                player_colors.append(player_color)

        if len(player_colors) < 2:
            return False

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.assignment_mode = "color"
        self.team_feature_centers = {0: kmeans.cluster_centers_[0], 1: kmeans.cluster_centers_[1]}
        self.cluster_to_team = {0: 1, 1: 2}
        self.team_cluster_ids = (0, 1)
        self._set_display_colors()
        return True

    def assign_team_color_from_colors(self, player_colors):
        """
        Fit the team KMeans from a pre-collected list of player color vectors.
        Use this when collecting colors from multiple frames for robustness.
        """
        if len(player_colors) < 2:
            return False

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.assignment_mode = "color"
        self.team_feature_centers = {0: kmeans.cluster_centers_[0], 1: kmeans.cluster_centers_[1]}
        self.cluster_to_team = {0: 1, 1: 2}
        self.team_cluster_ids = (0, 1)
        self._set_display_colors()
        return True

    def assign_team_from_embeddings(self, player_features):
        """
        Fit a 3-cluster KMeans over CAE embeddings and map the two largest
        clusters to Team 1 and Team 2. The remaining cluster acts as an
        outlier bucket.
        """
        if len(player_features) < 3:
            return False

        player_features = np.asarray(player_features, dtype=np.float32)
        kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_features)

        counts = np.bincount(kmeans.labels_, minlength=3)
        ordered_clusters = np.argsort(counts)[::-1]
        team_1_cluster = int(ordered_clusters[0])
        team_2_cluster = int(ordered_clusters[1])
        other_cluster = int(ordered_clusters[2]) if len(ordered_clusters) > 2 else None

        self.kmeans = kmeans
        self.assignment_mode = "embedding"
        self.team_feature_centers = {
            idx: center for idx, center in enumerate(kmeans.cluster_centers_)
        }
        self.cluster_to_team = {
            team_1_cluster: 1,
            team_2_cluster: 2,
        }
        self.team_cluster_ids = (team_1_cluster, team_2_cluster)
        if other_cluster is not None:
            self.cluster_to_team[other_cluster] = 0

        self._set_display_colors()
        return True

    def bootstrap_from_video(self, frames, player_tracks, sample_every=5, max_frames=150):
        sampled_features = []
        sampled_colors = []

        max_index = min(len(frames), len(player_tracks), max_frames)
        use_encoder = self.load_encoder()

        for frame_idx in range(max_index):
            if frame_idx % sample_every != 0:
                continue

            frame = frames[frame_idx]
            detections = player_tracks[frame_idx]

            if use_encoder:
                frame_crops = []
                for _, detection in detections.items():
                    bbox = detection.get("bbox")
                    if bbox is None:
                        continue
                    crop = self.get_player_crop(frame, bbox)
                    if crop is not None:
                        frame_crops.append(crop)

                if frame_crops:
                    batch = np.asarray(frame_crops, dtype=np.float32)
                    features_batch = self.encoder.predict(batch, verbose=0)
                    for features in features_batch:
                        sampled_features.append(np.asarray(features, dtype=np.float32).reshape(-1))
            else:
                for _, detection in detections.items():
                    bbox = detection.get("bbox")
                    if bbox is None:
                        continue
                    color = self.get_player_color(frame, bbox)
                    if color is not None:
                        sampled_colors.append(color)

        if use_encoder and len(sampled_features) >= 3:
            return self.assign_team_from_embeddings(sampled_features)

        if len(sampled_colors) >= 2:
            return self.assign_team_color_from_colors(sampled_colors)

        return False

    def _predict_team_from_embedding(self, embedding):
        if self.kmeans is None or embedding is None:
            return None

        raw_cluster = int(self.kmeans.predict(embedding.reshape(1, -1))[0])
        mapped_team = self.cluster_to_team.get(raw_cluster)
        if mapped_team in (1, 2):
            return mapped_team

        if len(self.kmeans.cluster_centers_) < 2:
            return 1

        team_1_cluster, team_2_cluster = self.team_cluster_ids
        team_1_center = self.team_feature_centers.get(team_1_cluster)
        team_2_center = self.team_feature_centers.get(team_2_cluster)
        if team_1_center is None or team_2_center is None:
            return 1

        dist_1 = np.linalg.norm(embedding - np.asarray(team_1_center, dtype=np.float32))
        dist_2 = np.linalg.norm(embedding - np.asarray(team_2_center, dtype=np.float32))
        return 1 if dist_1 <= dist_2 else 2

    def assign_teams_for_frame(self, frame, player_detections, frame_idx=None):
        """
        Assign teams for all player detections in a frame.
        In embedding mode this performs one batched encoder call per frame.
        """
        if self.kmeans is None:
            return {}

        team_ids = {}

        if self.assignment_mode != "embedding":
            for player_id, detection in player_detections.items():
                bbox = detection.get("bbox")
                if bbox is None:
                    continue
                team_ids[player_id] = self.get_player_team(frame, bbox, player_id, frame_idx=frame_idx)
            return team_ids

        to_eval_ids = []
        to_eval_crops = []

        for player_id, detection in player_detections.items():
            if player_id in self.player_team_dict:
                team_ids[player_id] = self.player_team_dict[player_id]
                continue

            if frame_idx is not None and player_id in self._last_eval_frame:
                if frame_idx - self._last_eval_frame[player_id] < self.reassign_interval:
                    cached_raw = self._last_raw_team.get(player_id)
                    team_ids[player_id] = self._update_and_get_stable_team(player_id, cached_raw)
                    continue

            bbox = detection.get("bbox")
            if bbox is None:
                team_ids[player_id] = self._update_and_get_stable_team(player_id, None)
                continue

            crop = self.get_player_crop(frame, bbox)
            if crop is None:
                team_ids[player_id] = self._update_and_get_stable_team(player_id, None)
                continue

            to_eval_ids.append(player_id)
            to_eval_crops.append(crop)

        if to_eval_crops:
            if self.encoder is None and not self.load_encoder():
                for player_id in to_eval_ids:
                    team_ids[player_id] = self._update_and_get_stable_team(player_id, None)
            else:
                feature_batch = self.encoder.predict(np.asarray(to_eval_crops, dtype=np.float32), verbose=0)
                for idx, player_id in enumerate(to_eval_ids):
                    embedding = np.asarray(feature_batch[idx], dtype=np.float32).reshape(-1)
                    raw_team = self._predict_team_from_embedding(embedding)
                    self._last_raw_team[player_id] = raw_team
                    if frame_idx is not None:
                        self._last_eval_frame[player_id] = frame_idx
                    team_ids[player_id] = self._update_and_get_stable_team(player_id, raw_team)

        for player_id in player_detections:
            if player_id not in team_ids:
                team_ids[player_id] = self._update_and_get_stable_team(player_id, None)

        return team_ids

    def get_player_team(self, frame, player_bbox, player_id, frame_idx=None):
        """
        Return the team id (1 or 2) for a given player.

        Uses a sliding window of the last *stability_window* raw votes.
        A new assignment is only accepted (or an existing one changed)
        when *all* votes in the window agree on the same team.
        Until that threshold is met the previously locked value is kept.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        raw_team = None

        if self.kmeans is not None:
            if frame_idx is not None and player_id in self._last_eval_frame:
                if frame_idx - self._last_eval_frame[player_id] < self.reassign_interval:
                    raw_team = self._last_raw_team.get(player_id)
                    return self._update_and_get_stable_team(player_id, raw_team)

            if self.assignment_mode == "embedding":
                player_feature = self.extract_player_features(frame, player_bbox)
                raw_team = self._predict_team_from_embedding(player_feature)
            else:
                player_color = self.get_player_color(frame, player_bbox)
                if player_color is not None:
                    raw_cluster = int(self.kmeans.predict(player_color.reshape(1, -1))[0])
                    raw_team = self.cluster_to_team.get(raw_cluster, raw_cluster + 1)

        if frame_idx is not None:
            self._last_eval_frame[player_id] = frame_idx
        self._last_raw_team[player_id] = raw_team

        return self._update_and_get_stable_team(player_id, raw_team)
