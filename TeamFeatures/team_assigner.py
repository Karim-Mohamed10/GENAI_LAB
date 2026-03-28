from sklearn.cluster import KMeans
from collections import defaultdict, deque
import numpy as np


class TeamAssigner:
    def __init__(self, stability_window=5):
        """
        Parameters
        ----------
        stability_window : int
            Number of consecutive frames that must agree on a team
            before the assignment is accepted or changed.
        """
        self.team_colors = {}
        self.player_team_dict = {}          # locked team assignment per player
        self.kmeans = None
        self.stability_window = stability_window

        # Per-player sliding window of raw per-frame votes
        self._vote_history = defaultdict(lambda: deque(maxlen=stability_window))

    def get_clustering_model(self, image):
        """Reshape image to 2D and perform K-means with 2 clusters."""
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        """Extract dominant player color from the top-half of the bounding box."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Clamp to frame boundaries
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        image = frame[y1:y2, x1:x2]

        # Skip if the crop is too small
        if image.shape[0] < 4 or image.shape[1] < 4:
            return None

        top_half_image = image[0:int(image.shape[0] / 2), :]

        if top_half_image.shape[0] < 2 or top_half_image.shape[1] < 2:
            return None

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster (non-background)
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Determine the two team colors by clustering the dominant colors
        of all detected players in a single frame.

        Parameters
        ----------
        frame : np.ndarray
            The video frame (BGR/RGB image).
        player_detections : dict
            {track_id: {"bbox": [x1, y1, x2, y2], ...}, ...}
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            if player_color is not None:
                player_colors.append(player_color)

        if len(player_colors) < 2:
            return False

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        return True

    def assign_team_color_from_colors(self, player_colors):
        """
        Fit the team KMeans from a pre-collected list of player color vectors.
        Use this when collecting colors from multiple frames for robustness.
        """
        if len(player_colors) < 2:
            return False

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        return True

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Return the team id (1 or 2) for a given player.

        Uses a sliding window of the last *stability_window* raw votes.
        A new assignment is only accepted (or an existing one changed)
        when *all* votes in the window agree on the same team.
        Until that threshold is met the previously locked value is kept.
        """
        # --- get this frame's raw vote ---
        player_color = self.get_player_color(frame, player_bbox)

        if player_color is not None:
            raw_team = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        else:
            raw_team = None

        # push vote (None means "no opinion this frame")
        if raw_team is not None:
            self._vote_history[player_id].append(raw_team)

        history = self._vote_history[player_id]

        # --- decide whether to lock / change ---
        if len(history) >= self.stability_window:
            if len(set(history)) == 1:
                consensus = history[0]
                self.player_team_dict[player_id] = consensus

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if history:
            return max(set(history), key=list(history).count)
        return 1  # ultimate fallback
