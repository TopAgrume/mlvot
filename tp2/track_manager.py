class Track:
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.iou_score = None
        self.bbox = bbox
        self.age = 1
        self.consecutive_invisible_count = 0
        self.history = [(self.age, bbox)]
        self.is_active = True

    def update(self, bbox, iou_score):
        self.bbox = bbox
        self.iou_score = iou_score
        self.age += 1
        self.consecutive_invisible_count = 0
        self.history.append((self.age, bbox))

    def mark_invisible(self):
        self.age += 1
        self.consecutive_invisible_count += 1

class TrackManager:
    def __init__(self, max_invisible):
        self.max_invisible = max_invisible
        self.tracks = []

    def create_track(self, bbox, track_id):
        track = Track(bbox, track_id)
        self.tracks.append(track)
        return track

    def get_active_tracks(self):
        return [track for track in self.tracks if track.is_active]

    def update_unmatched_track(self, track):
        track.mark_invisible()
        if track.consecutive_invisible_count > self.max_invisible:
            track.is_active = False

    def update_unmatched_tracks(self):
        for track in self.get_active_tracks():
            self.update_unmatched_track(track)