class Track:
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.bbox = bbox
        self.appearance = None
        self.age = 1
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
        self.history = [(self.age, bbox)]
        self.is_active = True
        self.kf = None
        self.reid_features = None

    def update(self, bbox, appearance):
        self.bbox = bbox
        self.appearance = appearance
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
        self.history.append((self.age, bbox))

    def mark_invisible(self):
        self.age += 1
        self.consecutive_invisible_count += 1

class TrackManager:
    def __init__(self, max_age):
        self.max_age = max_age
        self.tracks = []

    def create_track(self, bbox, track_id):
        track = Track(bbox, track_id)
        self.tracks.append(track)
        return track

    def get_active_tracks(self):
        return [track for track in self.tracks if track.is_active]

    def update_unmatched_track(self, track):
        track.mark_invisible()
        if track.consecutive_invisible_count > self.max_age:
            track.is_active = False

    def update_unmatched_tracks(self):
        for track in self.get_active_tracks():
            self.update_unmatched_track(track)