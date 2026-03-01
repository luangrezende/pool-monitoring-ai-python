import cv2
from modules.pool_boundary import PoolBoundary
from modules.point_selector import PointSelector, save_coordinates, load_coordinates
from modules.object_detector import ObjectDetector
from modules.config_manager import ConfigManager


def configure_pool_boundary(video_path, config):
    pool_coordinates = load_coordinates()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file!")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error reading video frame!")
        return None
    
    if pool_coordinates:
        print("Existing configuration found!")
        print("Press 'c' to configure new points, or any other key to use saved config...")
        
        preview = frame.copy()
        boundary = PoolBoundary(pool_coordinates, config)
        boundary.draw(preview)
        
        ui_config = config.get_ui_config()
        window_mode = cv2.WINDOW_NORMAL if ui_config['resizable_windows'] else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow('Current Configuration', window_mode)
        cv2.imshow('Current Configuration', preview)
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow('Current Configuration')
        
        hotkeys = config.get_hotkeys()
        if key != ord(hotkeys['reconfigure_boundary']):
            print("Using saved configuration")
            return pool_coordinates
    
    boundary_config = config.get_pool_boundary_config()
    print("\nInteractive Point Selection:")
    print(f"1. LEFT CLICK to add points (minimum {boundary_config['min_points']}, maximum {boundary_config['max_points']})")
    print("2. LEFT CLICK and DRAG to move points")
    print("3. RIGHT CLICK to remove a point")
    print("4. Press C to clear all points")
    print(f"5. Press ENTER when done (minimum {boundary_config['min_points']} points)")
    print("6. Press ESC to cancel")
    print("\nTip: Add as many points as needed for accurate pool boundary")
    
    selector = PointSelector(
        frame, 
        min_points=boundary_config['min_points'], 
        max_points=boundary_config['max_points'],
        config=config
    )
    pool_coordinates = selector.select_points()
    
    if pool_coordinates:
        save_coordinates(pool_coordinates)
        print("Pool boundary configured successfully!")
        return pool_coordinates
    else:
        print("Configuration cancelled")
        return None


def main():
    config = ConfigManager()
    
    video_path = config.get_video_path()
    enable_detection = config.get_detection_enabled()
    
    pool_coordinates = configure_pool_boundary(video_path, config)
    
    if not pool_coordinates:
        print("No pool configuration available. Exiting.")
        return
    
    pool_boundary = PoolBoundary(pool_coordinates, config)
    
    detector = None
    if enable_detection:
        print("\nInitializing YOLOv8 object detector...")
        try:
            detection_config = config.get_detection_config()
            detector = ObjectDetector(
                model_size=detection_config['model_size'],
                confidence_threshold=detection_config['confidence_threshold'],
                skip_frames=detection_config['skip_frames'],
                process_size=detection_config['process_size'],
                config=config
            )
            print("Object detection enabled!")
        except Exception as e:
            print(f"Warning: Could not load detector: {e}")
            print("Continuing without object detection...")
            enable_detection = False
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    ui_config = config.get_ui_config()
    window_name = config.get('video', 'display_window_name')
    window_mode = cv2.WINDOW_NORMAL if ui_config['resizable_windows'] else cv2.WINDOW_AUTOSIZE
    performance_config = config.get_performance_config()
    hotkeys = config.get_hotkeys()
    
    print(f"\nPool monitoring started.")
    print(f"Press '{hotkeys['quit']}' to quit")
    if enable_detection:
        print(f"Press '{hotkeys['toggle_detection']}' to toggle detection on/off")
    
    cv2.namedWindow(window_name, window_mode)
    
    detection_active = enable_detection
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        alert_info = None
        if detection_active and detector:
            frame, alert_info = detector.detect(frame, pool_boundary)
            if alert_info:
                frame = detector.draw_alerts(frame, alert_info)
        
        frame = pool_boundary.draw(frame)
        
        status_text = "Detection: ON" if detection_active else "Detection: OFF"
        status_color = (0, 255, 0) if detection_active else (128, 128, 128)
        
        cv2.putText(frame, status_text, (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, ui_config['font_scale'], status_color, ui_config['font_thickness'])
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(performance_config['video_frame_delay_ms']) & 0xFF
        if key == ord(hotkeys['quit']):
            break
        elif key == ord(hotkeys['toggle_detection']) and enable_detection:
            detection_active = not detection_active
            state = "enabled" if detection_active else "disabled"
            print(f"Object detection {state}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Pool monitoring stopped.")


if __name__ == "__main__":
    main()
