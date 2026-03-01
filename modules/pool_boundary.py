import cv2
import numpy as np

class PoolBoundary:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates, dtype=np.int32)
        self.color = (0, 255, 0)
        self.thickness = 2
    
    def draw(self, frame):
        cv2.polylines(frame, [self.coordinates], True, self.color, self.thickness)
        return frame
