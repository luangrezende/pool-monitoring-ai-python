import cv2
import numpy as np
import json


class PointSelector:
    
    def __init__(self, frame, min_points=3, max_points=20, config=None):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.min_points = min_points
        self.max_points = max_points
        self.points = []
        self.dragging_index = None
        self.hover_index = None
        self.config = config
        
        if config:
            selector_config = config.get_point_selector_config()
            self.point_radius = selector_config['point_radius']
            self.hover_radius = selector_config['hover_radius']
            self.window_name = selector_config['window_name']
        else:
            self.point_radius = 8
            self.hover_radius = 12
            self.window_name = 'Select Pool Boundary Points'
        
    def mouse_callback(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_index = None
            for i, point in enumerate(self.points):
                distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                if distance < self.hover_radius:
                    self.hover_index = i
                    break
            
            if self.dragging_index is not None:
                self.points[self.dragging_index] = [x, y]
            
            self.update_display()
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.points):
                distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                if distance < self.hover_radius:
                    self.dragging_index = i
                    return
            
            if len(self.points) < self.max_points:
                self.points.append([x, y])
                self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_index = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                min_distance = float('inf')
                closest_index = None
                for i, point in enumerate(self.points):
                    distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                    if distance < min_distance and distance < self.hover_radius * 2:
                        min_distance = distance
                        closest_index = i
                
                if closest_index is not None:
                    self.points.pop(closest_index)
                    self.update_display()
    
    def update_display(self):
        self.frame = self.original_frame.copy()
        
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(self.frame, [pts], True, (0, 255, 0), 2)
            overlay = self.frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, self.frame, 0.8, 0, self.frame)
        
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.frame, tuple(self.points[i]), 
                        tuple(self.points[i + 1]), (255, 255, 0), 1)
        
        for i, point in enumerate(self.points):
            color = (0, 255, 255) if i == self.hover_index else (0, 0, 255)
            radius = self.hover_radius if i == self.hover_index else self.point_radius
            cv2.circle(self.frame, tuple(point), radius, color, -1)
            cv2.circle(self.frame, tuple(point), radius + 2, (255, 255, 255), 2)
            cv2.putText(self.frame, str(i + 1), (point[0] - 5, point[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        instructions = [
            "LEFT CLICK: Add/Drag point",
            "RIGHT CLICK: Remove point",
            f"Points: {len(self.points)} (min: {self.min_points}, max: {self.max_points})",
            "Press C to clear | ENTER to save | ESC to cancel"
        ]
        y_offset = 30
        for instruction in instructions:
            cv2.putText(self.frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        cv2.imshow(self.window_name, self.frame)
    
    def select_points(self):
        window_mode = cv2.WINDOW_NORMAL if self.config and self.config.get('ui', 'resizable_windows') else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(self.window_name, window_mode)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:
                if len(self.points) >= self.min_points:
                    cv2.destroyWindow(self.window_name)
                    return self.points
                else:
                    print(f"Please select at least {self.min_points} points! (Current: {len(self.points)})")
            
            elif key == 27:
                cv2.destroyWindow(self.window_name)
                return None
            
            elif key == ord('c'):
                self.points = []
                self.update_display()


def save_coordinates(coordinates, filename='pool_config.json'):
    config = {
        'pool_coordinates': coordinates
    }
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filename}")


def load_coordinates(filename='pool_config.json'):
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        return config['pool_coordinates']
    except FileNotFoundError:
        return None
