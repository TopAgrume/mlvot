import numpy as np
from scipy.optimize import linear_sum_assignment
from tp2.track_manager import TrackManager
from tp2.visualizer import TrackVisualizer

class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_invisible=10):
        self.iou_threshold = iou_threshold                  # Min threshold when handling matched tracks and detections
        self.max_invisible = max_invisible                  # Number of unmatched iterations before deactivating the tracker
        self.track_manager = TrackManager(max_invisible)    # To handle tracks easily
        self.next_id = 1                                    # ID for new box trackers
        self.visualizer = TrackVisualizer()                 # Usefull to represent boxes on the video


    def compute_iou(self, box1, box2):
        x_1, y_1, w_1, h_1 = box1
        x_2, y_2, w_2, h_2 = box2

        x_left = max(x_1, x_2)
        y_top = max(y_1, y_2)
        x_right = min(x_1 + w_1, x_2 + w_2)
        y_bot = min(y_1 + h_1, y_2 + h_2)

        if x_right < x_left or y_bot < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bot - y_top)
        union = w_1 * h_1 + w_2 * h_2 - intersection

        return intersection / union if union > 0 else 0


    def create_similarity_matrix(self, tracks, detections):
        similarity_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                similarity_matrix[i, j] = self.compute_iou(track.bbox, detection[2:6])

        return similarity_matrix


    def update(self, detections, frame):
        tracks = self.track_manager.get_active_tracks()

        if len(tracks) == 0:
            # Add new tracks for all detections
            for detection in detections:
                self.track_manager.create_track(detection[2:6], self.next_id)
                self.next_id += 1
            return

        if len(detections) == 0:
            self.track_manager.update_unmatched_tracks()
            return

        similarity_matrix = self.create_similarity_matrix(tracks, detections)

        # Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(-similarity_matrix)

        matched_track_indices = []
        matched_detection_indices = []

        # Matched tracks and detections
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if similarity_matrix[track_idx, detection_idx] >= self.iou_threshold:
                matched_track_indices.append(track_idx)
                matched_detection_indices.append(detection_idx)
                tracks[track_idx].update(detections[detection_idx][2:6], similarity_matrix[track_idx, detection_idx])

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detection_indices]

        # Unmatched tracks
        for track_idx in unmatched_tracks:
            self.track_manager.update_unmatched_track(tracks[track_idx])

        # Unmatched detections -> init new track
        for detection_idx in unmatched_detections:
            self.track_manager.create_track(detections[detection_idx][2:6], self.next_id)
            self.next_id += 1

        self.visualizer.draw_tracking_results(frame, tracks)

        return tracks


    def save_results(self, filename, tracks):
        with open(filename, 'w') as f:
            for track in tracks:
                for frame, bbox in track.history:
                    x, y, w, h = bbox
                    f.write(f"{frame},{track.id},{x},{y},{w},{h},1,-1,-1,-1\n")