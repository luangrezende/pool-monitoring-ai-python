from ultralytics import YOLO
import cv2
import numpy as np
import time


class ObjectDetector:
    
    def __init__(self, model_size='n', confidence_threshold=0.5, skip_frames=2, process_size=640, config=None):
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        self.skip_frames = skip_frames
        self.process_size = process_size
        self.frame_count = 0
        self.config = config
        
        detection_config = config.get_detection_config()
        self.min_area_percent = detection_config['min_detection_area_percent']
        self.max_area_percent = detection_config['max_detection_area_percent']
        self.alert_delay_seconds = detection_config['alert_delay_seconds']
        self.grace_period = detection_config['grace_period_seconds']
        self.person_confidence_threshold = detection_config['person_confidence_threshold']
        
        messages = detection_config['messages']
        self.msg_alert_person = messages['alert_person']
        self.msg_alert_object = messages['alert_object']
        self.msg_alert_both = messages['alert_both']
        
        display_options = detection_config['display']
        self.show_timer_on_boxes = display_options['show_timer_on_boxes']
        self.show_counter_before_alert = display_options['show_counter_before_alert']
        
        self.zone_occupied_since = None
        self.last_detection_time = None
        self.tracked_objects = {}
        self.track_classifications = {}
        self.max_age = 10
        
        alert_config = config.get_alert_box_config()
        self.color_in_pool = alert_config['color_in_pool']
        self.color_outside = alert_config['color_outside']
        self.thickness_in_pool = alert_config['thickness_in_pool']
        self.thickness_outside = alert_config['thickness_outside']
        
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("Model loaded successfully!")
        
        self.PERSON_CLASS_ID = 0
        
    def detect(self, frame, pool_boundary=None):
        self.frame_count += 1
        current_time = time.time()
        
        if self.frame_count % self.skip_frames == 0:
            orig_height, orig_width = frame.shape[:2]
            
            scale = self.process_size / orig_width
            if scale < 1.0:
                process_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                process_frame = frame
                scale = 1.0
            
            results = self.model.track(
                process_frame, 
                conf=self.confidence_threshold, 
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml"
            )
            
            detections_in_pool = []
            
            for result in results:
                boxes = result.boxes
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
                
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    
                    detection_width = x2 - x1
                    detection_height = y2 - y1
                    detection_area = detection_width * detection_height
                    frame_area = orig_width * orig_height
                    area_percent = (detection_area / frame_area) * 100
                    
                    if area_percent < self.min_area_percent or area_percent > self.max_area_percent:
                        continue
                    
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    track_id = int(track_ids[idx]) if track_ids is not None else None
                    
                    if track_id is not None and track_id in self.track_classifications:
                        is_person = self.track_classifications[track_id]
                    else:
                        is_person = (class_id == self.PERSON_CLASS_ID)
                        
                        if is_person and confidence < self.person_confidence_threshold:
                            is_person = False
                        
                        if track_id is not None:
                            self.track_classifications[track_id] = is_person
                    
                    object_type = 'PERSON' if is_person else 'OBJECT'
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    in_pool = False
                    if pool_boundary:
                        in_pool = pool_boundary.is_point_inside((center_x, center_y))
                    
                    if in_pool:
                        detection_data = {
                            'type': object_type,
                            'is_person': is_person,
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'track_id': track_id,
                            'age': 0
                        }
                        detections_in_pool.append(detection_data)
                        
                        if track_id is not None:
                            self.tracked_objects[track_id] = detection_data
            
            current_track_ids = set(d.get('track_id') for d in detections_in_pool if d.get('track_id') is not None)
            objects_to_remove = []
            
            for track_id in list(self.tracked_objects.keys()):
                if track_id not in current_track_ids:
                    self.tracked_objects[track_id]['age'] += 1
                    if self.tracked_objects[track_id]['age'] > self.max_age:
                        objects_to_remove.append(track_id)
                    else:
                        detections_in_pool.append(self.tracked_objects[track_id])
            
            for track_id in objects_to_remove:
                del self.tracked_objects[track_id]
                if track_id in self.track_classifications:
                    del self.track_classifications[track_id]
            
            self.last_pool_detections = detections_in_pool
            self.last_detection_update_time = current_time
        else:
            detections_in_pool = self.last_pool_detections if hasattr(self, 'last_pool_detections') else []
            
            for track_id in self.tracked_objects:
                self.tracked_objects[track_id]['age'] += 1
        
        zone_has_detections = len(detections_in_pool) > 0
        
        if zone_has_detections:
            self.last_detection_time = current_time
            
            if self.zone_occupied_since is None:
                self.zone_occupied_since = current_time
            
            time_in_zone = current_time - self.zone_occupied_since
        else:
            if self.last_detection_time is not None:
                time_since_last_detection = current_time - self.last_detection_time
                
                if time_since_last_detection > self.grace_period:
                    self.zone_occupied_since = None
                    self.last_detection_time = None
                    time_in_zone = 0
                else:
                    time_in_zone = current_time - self.zone_occupied_since if self.zone_occupied_since else 0
            else:
                time_in_zone = 0
        
        show_alert = time_in_zone >= self.alert_delay_seconds
        
        for detection in detections_in_pool:
            x1, y1, x2, y2 = detection['box']
            object_type = detection['type']
            
            if show_alert:
                color = self.color_in_pool
                thickness = self.thickness_in_pool
            else:
                color = (255, 165, 0)
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            if self.show_timer_on_boxes:
                label = f'{object_type} {time_in_zone:.1f}s'
            else:
                label = object_type
                
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            
            cv2.rectangle(frame, 
                        (x1, label_y - label_size[1] - 5),
                        (x1 + label_size[0], label_y + 5),
                        color, -1)
            cv2.putText(frame, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame, {
            'show_alert': show_alert,
            'time_in_zone': time_in_zone,
            'zone_occupied': self.zone_occupied_since is not None,
            'detection_count': len(detections_in_pool),
            'has_person': any(d['is_person'] for d in detections_in_pool) if len(detections_in_pool) > 0 else False,
            'has_object': any(not d['is_person'] for d in detections_in_pool) if len(detections_in_pool) > 0 else False
        }
    
    def draw_alerts(self, frame, alert_info):
        time_in_zone = alert_info['time_in_zone']
        zone_occupied = alert_info['zone_occupied']
        
        if zone_occupied and not alert_info['show_alert'] and self.show_counter_before_alert:
            timer_text = f"Zone occupied: {time_in_zone:.1f}s / {self.alert_delay_seconds:.1f}s"
            
            text_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (15 + text_size[0], 35), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            cv2.putText(frame, timer_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        if alert_info['show_alert']:
            alert_height = 70
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], alert_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            if alert_info['has_person'] and alert_info['has_object']:
                alert_text = self.msg_alert_both
            elif alert_info['has_person']:
                alert_text = self.msg_alert_person
            else:
                alert_text = self.msg_alert_object
            
            cv2.putText(frame, alert_text, (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return frame
