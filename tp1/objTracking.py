from Detector import detect
from KalmanFilter import KalmanFilter
import cv2
import numpy as np

kalman_filter = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1)

video_path = "data/randomball.avi"
cap = cv2.VideoCapture(video_path)
trajectory = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of the video")
        break

    centers = detect(frame=frame)

    if len(centers) > 0:
        # ===== Prediction and estimation =====
        kalman_filter.predict()
        pred_x, pred_y = int(kalman_filter.x[0].item()), int(kalman_filter.x[1].item())

        kalman_filter.update(centers[0])
        estim_x, estim_y = int(kalman_filter.x[0].item()), int(kalman_filter.x[1].item())

        trajectory.append((estim_x, estim_y))

        # ===== Visualize tracking results =====
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(img=frame, pt1=trajectory[i-1], pt2=trajectory[i],
                        color=(255, 0, 255), thickness=2)

        # Predicted positions
        cv2.rectangle(img=frame,
            pt1=(pred_x - 10, pred_y - 10),
            pt2=(pred_x + 10, pred_y + 10),
            color=(0, 0, 255),
            thickness=2
        )

        # Estimated positions
        cv2.rectangle(img=frame,
            pt1=(estim_x - 10, estim_y - 10),
            pt2=(estim_x + 10, estim_y + 10),
            color=(255, 0, 0),
            thickness=2
        )

        # Detected circle
        cv2.circle(img=frame, center=(int(centers[0][0].item()),int(centers[0][1].item())),
                   radius=10, color=(0, 255, 0), thickness=2)


    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
