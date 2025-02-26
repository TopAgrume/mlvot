import cv2
import numpy as np
from collections import defaultdict
from tp4.appearance_kalman_tracker import AppearanceKalmanTracker

def load_detections(detection_file):
    detections = defaultdict(list)
    with open(detection_file, 'r') as f:
        for line in f:
            frame, _, left, top, width, height, conf, x, y, z = map(float, line.split(','))
            frame_id = int(frame)
            detections[frame_id].append([frame, -1, left, top, width, height, conf, x, y, z])
    return detections

def main():
    video = cv2.VideoCapture("ADL-Rundle-6/img1/%06d.jpg")

    tracker = AppearanceKalmanTracker(
        iou_threshold=0.3,
        max_invisible=10,
        combination_weight=0.5
    )

    detections = load_detections("ADL-Rundle-6/det/public-dataset/det.txt")

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        filename='tp4/results/tracking_result.avi',
        fourcc=cv2.VideoWriter_fourcc(*'XVID'),
        fps=30,
        frameSize=(frame_width, frame_height)
    )

    frame_idx = 1
    tracks = []

    while True:
        ret, frame = video.read()
        if not ret:
            print("End of the video")
            break

        frame_detections = detections.get(frame_idx, [])
        current_tracks = tracker.update(frame, frame_detections)

        if current_tracks:
            tracks.extend(current_tracks)

        out.write(frame)

        cv2.imshow('TP4 Tracking with Appearance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    tracker.save_results("tp4/results/ADL-Rundle-6_tracking_results_appearance.txt", tracks)

    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()