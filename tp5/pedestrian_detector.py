import torch
import cv2

class PedestrianDetector:
    def __init__(self, model_name='yolov5s', conf_thresh=0.3):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.model.conf = conf_thresh
        self.model.classes = [0]  # class 0 in COCO for pedestrians
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, frame, frame_idx):
        results = self.model(frame)
        detections = []

        for *xyxy, conf, _ in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            detections.append([frame_idx, -1, x1, y1, w, h, float(conf), -1, -1, -1])

        return detections

    def save_detections(self, video_path, output_file):
        frame_idx = 1
        video = cv2.VideoCapture(video_path)

        with open(output_file, 'w') as f:
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                detections = self.detect(frame, frame_idx)

                for det in detections:
                    f.write(f"{','.join(map(str, det))}\n")

                frame_idx += 1

        video.release()