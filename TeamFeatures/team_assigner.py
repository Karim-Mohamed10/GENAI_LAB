from sklearn.cluster import KMeans
from collections import defaultdict, deque
import numpy as np
import cv2


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
        """Extract dominant player color by masking out grass and normalizing brightness."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h_frame, w_frame = frame.shape[:2]

        # 1. ULTRA-TIGHT CROP (Avoid shorts, hair, and ARMS/SKIN)
        h_box, w_box = y2 - y1, x2 - x1
        crop_y1 = max(0, y1 + int(h_box * 0.15))
        crop_y2 = min(h_frame, y1 + int(h_box * 0.55))
        crop_x1 = max(0, x1 + int(w_box * 0.30))  # Squeezed from 0.20 to 0.30 to ignore arms
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

        # 5. NORMALIZE BRIGHTNESS (The Shadow Fix)
        # Convert extracted pixels to HSV, force constant high brightness, and convert back
        pixels_reshaped = player_pixels.reshape(-1, 1, 3)
        hsv_pixels = cv2.cvtColor(pixels_reshaped, cv2.COLOR_BGR2HSV)
        hsv_pixels[:, :, 2] = 200  # Set 'Value' (brightness) to a constant high value
        normalized_bgr = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR).reshape(-1, 3)

        # 6. FIND TRUE DOMINANT COLOR
        # Run KMeans on the shadow-free BGR values
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
