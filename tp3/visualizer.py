import cv2
import numpy as np

class TrackVisualizer:
    def __init__(self):
        self.colors = {}
        self.trace_length = 30

    def get_color(self, track_id):
        if track_id not in self.colors:
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]

    def draw_bbox(self, frame, bbox, color, thickness=2):
        x, y, w, h = map(int, bbox)
        cv2.rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=color,
            thickness=thickness
        )

    def draw_kalman_state(self, frame, track):
        if track.kf is None:
            return

        color = self.get_color(track.id)
        current_x, current_y = int(track.kf.x[0].item()), int(track.kf.x[1].item())

        # Draw centroid
        cv2.circle(
            img=frame,
            center=(current_x, current_y),
            radius=5,
            color=color,
            thickness=-1
        )

        # Draw velocity vector
        if len(track.kf.x) >= 4:
            end_x = int(current_x + track.kf.x[2].item())
            end_y = int(current_y + track.kf.x[3].item())

            cv2.arrowedLine(
                img=frame,
                pt1=(current_x, current_y),
                pt2=(end_x, end_y),
                color=color,
                thickness=2,
                tipLength=0.3
            )


    def draw_history_trace(self, frame, track, max_history=30):
        if len(track.history) < 2:
            return

        color = self.get_color(track.id)

        points = []
        history = track.history[-max_history:]

        for _, bbox in history:
            x, y, w, h = bbox
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            points.append((center_x, center_y))

        # History lines -> connecting the points
        for i in range(1, len(points)):
            cv2.line(
                img=frame,
                pt1=points[i-1],
                pt2=points[i],
                color=color,
                thickness=2
            )

        # Draw history dots
        for point in points:
            cv2.circle(
                img=frame,
                center=point,
                radius=3,
                color=color,
                thickness=-1
            )

    def draw_tracking_results(self, frame, tracks):
        for track in tracks:
            if not track.is_active:
                continue

            x, y, w, h = map(int, track.bbox)
            color = self.get_color(track.id)

            # bounding box
            self.draw_bbox(frame, (x, y, w, h), color)

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

            self.draw_kalman_state(frame, track)
            self.draw_history_trace(frame, track, self.trace_length)

        return frame


    def save_frame(self, frame, output_path):
        cv2.imwrite(output_path, frame)