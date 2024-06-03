from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = bbox.reshape((4, 1))

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque(maxlen=30)
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.active = True

    def update(self, bbox):
        self.time_since_update = 0
        self.history.append(bbox)
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox.reshape((4, 1)))

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.kf.x[:4].reshape((1, 4)))
        return self.kf.x

    def get_state(self):
        return self.kf.x