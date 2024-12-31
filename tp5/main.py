import cv2
from collections import defaultdict
from tp5.pedestrian_detector import PedestrianDetector
from tp4.appearance_kalman_tracker import AppearanceKalmanTracker
from metrics import Metrics
import os
import warnings
warnings.filterwarnings("ignore")

def load_detections(detection_file):
    detections = defaultdict(list)
    with open(detection_file, 'r') as f:
        for line in f:
            frame, _, left, top, width, height, conf, x, y, z = map(float, line.split(','))
            frame_id = int(frame)
            detections[frame_id].append([frame, -1, left, top, width, height, conf, x, y, z])
    return detections

def main():
    # ==== DEFAULT VALUES CONFIG =====
    video_path = "ADL-Rundle-6/img1/%06d.jpg"
    gt_folder = "ADL-Rundle-6/gt"

    base_dir = "tp5"
    results_dir = os.path.join(base_dir, "results")
    output_det_path = os.path.join(results_dir, "det.txt")
    tracking_results_dir = os.path.join(base_dir, "results")
    output_tracking_path = os.path.join(tracking_results_dir, "ADL-Rundle-6_better_tracking_results_appearance.txt")


    # >>> Generating detections using yolov5s
    print("\n>>> Running PedestrianDetector")
    detector = PedestrianDetector(conf_thresh=0.3)
    detector.save_detections(video_path, output_det_path)

    # >>> Run AppearanceKalman using the new detections
    print("\n>>> Running AppearanceKalmanTracker")
    video = cv2.VideoCapture(video_path)

    tracker = AppearanceKalmanTracker(
        iou_threshold=0.3,
        max_invisible=10,
        combination_weight=0.5
    )

    detections = load_detections(output_det_path)

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        filename=os.path.join(results_dir, 'tracking_result.avi'),
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

        cv2.imshow('TP5 Tracking with YOLOv5 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    tracker.save_results(output_tracking_path, tracks)

    video.release()
    out.release()
    cv2.destroyAllWindows()

    # >>> Evaluate tracking performance
    #print("\nEvaluating tracking performance")
    #evaluator = Metrics(
    #    gt_folder=gt_folder,
    #    tracker_folder=base_dir,
    #    seq_name="ADL-Rundle-6"
    #)
    #metrics = evaluator.evaluate()

if __name__ == "__main__":
    main()