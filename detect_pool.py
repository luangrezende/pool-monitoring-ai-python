import cv2
from modules.pool_boundary import PoolBoundary

def main():
    # Configuration
    video_path = "pool.mp4"
    
    # Define the 4 pool coordinates (x, y) - top-left, top-right, bottom-right, bottom-left
    pool_coordinates = [
        [480, 350],  # Top-left corner
        [1440, 560],  # Top-right corner
        [1250, 980],  # Bottom-right corner
        [60, 450]    # Bottom-left corner
    ]
    
    pool_boundary = PoolBoundary(pool_coordinates)
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    print("Pool boundary detection started. Press 'q' to quit.")
    
    while True:
        # Read the frame
        ret, frame = cap.read()
        
        # If unable to read frame (end of video), restart
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Draw pool boundary
        frame = pool_boundary.draw(frame)
        
        # Display the frame
        cv2.imshow('Pool Boundary Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Pool boundary detection stopped.")

if __name__ == "__main__":
    main()
