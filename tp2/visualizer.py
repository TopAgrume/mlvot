import cv2
import numpy as np

class TrackVisualizer:
    def __init__(self):
        self.colors = {}

    def get_color(self, track_id):
        if track_id not in self.colors:
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]

    def draw_tracking_results(self, frame, tracks):
        for track in tracks:
            if not track.is_active:
                continue

            x, y, w, h = map(int, track.bbox)
            color = self.get_color(track.id)

            # bounding box
            cv2.rectangle(img=frame,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=color,
                thickness=2
            )

            # label on box
            info_text = f"ID: {track.id}" + f", IoU: {round(track.iou_score, 2)}" if track.iou_score else ""
            cv2.putText(img=frame,
                text=info_text,
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color,
                thickness=2
            )

        return frame

    def save_frame(self, frame, output_path):
        cv2.imwrite(output_path, frame)