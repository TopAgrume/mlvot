import numpy as np

class KalmanFilter:
    def __init__(self,
            dt: float,
            u_x: int,
            u_y: int,
            std_acc: int,
            x_sdt_meas: float,
            y_sdt_meas: float
        ):
        self.u = np.array([[u_x], [u_y]])
        self.x = np.zeros(shape=(4, 1))
        self.A = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]]
        )

        self.B = np.array(
            [[0.5 * dt ** 2, 0],
            [0, 0.5 * dt ** 2],
            [dt, 0],
            [0, dt]]
        )

        self.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]]
        )
        self.Q = np.array(
            [[0.25 * dt ** 4, 0, 0.5 * dt ** 3, 0],
             [0, 0.25 * dt ** 4, 0, 0.5 * dt ** 3],
             [0.5 * dt ** 3, 0, dt ** 2, 0],
             [0, 0.5 * dt ** 3, 0, dt ** 2]]
        ) * std_acc ** 2

        self.R = np.array(
            [[x_sdt_meas, 0],
             [0, y_sdt_meas]]
        )

        self.P = np.identity(self.A.shape[1])


    def predict(self):
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z_k: np.ndarray):
        S_k = self.H @ self.P @ self.H.T + self.R
        K_k = self.P @ self.H.T @ np.linalg.inv(S_k)

        y = z_k - self.H @ self.x
        self.x = self.x + K_k @ y
        self.P = (np.eye(self.P.shape[0]) - K_k @ self.H) @ self.P