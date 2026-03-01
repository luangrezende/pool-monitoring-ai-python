import cv2
import numpy as np

class PoolBoundary:
    def __init__(self, coordinates, config=None):
        self.coordinates = np.array(coordinates, dtype=np.int32)
        
        boundary_config = config.get_pool_boundary_config()
        self.color = boundary_config['color']
        self.thickness = boundary_config['thickness']
    
    def draw(self, frame):
        cv2.polylines(frame, [self.coordinates], True, self.color, self.thickness)
        return frame
    
    def is_point_inside(self, point):
        result = cv2.pointPolygonTest(self.coordinates, point, False)
        return result >= 0
