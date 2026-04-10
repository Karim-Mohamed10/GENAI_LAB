class GoalkeeperDetector:
    def __init__(self, left_box_limit=16.5, right_box_limit=88.5):
        self.left_box_limit = left_box_limit
        self.right_box_limit = right_box_limit

    def separate_goalkeepers(self, players_tracks, H, get_foot_fn, transform_fn):

        left_gk_id, right_gk_id = None, None
        min_x, max_x = self.left_box_limit, self.right_box_limit
        
        # 1. Find the deepest players
        for tid, d in list(players_tracks.items()):
            foot = get_foot_fn(d["bbox"])
            raw_pos = transform_fn(foot, H)
            if raw_pos is not None:
                if raw_pos[0] < min_x:
                    min_x = raw_pos[0]
                    left_gk_id = tid
                if raw_pos[0] > max_x:
                    max_x = raw_pos[0]
                    right_gk_id = tid
                    
        # 2. Extract them into a new dictionary
        goalkeepers_tracks = {}
        for gid in (left_gk_id, right_gk_id):
            if gid is not None and gid in players_tracks:
                goalkeepers_tracks[gid] = players_tracks.pop(gid)
                
        return goalkeepers_tracks
