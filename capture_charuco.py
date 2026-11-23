import threading
import cv2
import os

# Global variables for thread synchronization and frame storage
latest_frames = {}
frame_lock = threading.Lock()
stop_threads = False

def stream_camera(cam_source, cam_name):
    global latest_frames, stop_threads
    
    cap = cv2.VideoCapture(cam_source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_source}")
        return

    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {cam_source} failed to read frame.")
            break
            
        with frame_lock:
            latest_frames[cam_name] = frame
            
    cap.release()

def main():
    global stop_threads
    
    dir_cam0 = "cal_cam0"
    dir_cam1 = "cal_cam1"
    os.makedirs(dir_cam0, exist_ok=True)
    os.makedirs(dir_cam1, exist_ok=True)
    
    # Start background threads for live streaming
    t0 = threading.Thread(target=stream_camera, args=("/dev/video0", "cam0"))
    t1 = threading.Thread(target=stream_camera, args=("/dev/video2", "cam1"))
    
    t0.start()
    t1.start()
    
    print("Live previews active. Click on the preview windows and press [SPACE] to capture a pair, or [q] to quit.")
    
    capture_count = 0
    
    try:
        while True:
            with frame_lock:
                frame0 = latest_frames.get("cam0")
                frame1 = latest_frames.get("cam1")
                
            # Show separate pop-up windows for each camera
            if frame0 is not None:
                h, w, _ = frame0.shape
                preview0 = cv2.resize(frame0, (int(w / 2), int(h / 2)))
                cv2.imshow("Camera 0 (cal_cam0)", preview0)
                
            if frame1 is not None:
                h, w, _ = frame1.shape
                preview1 = cv2.resize(frame1, (int(w / 2), int(h / 2)))
                cv2.imshow("Camera 1 (cal_cam1)", preview1)
                
            key = cv2.waitKey(1) & 0xFF
            
            # Press SPACEBAR to capture the current frame pair simultaneously
            if key == ord(' '):
                with frame_lock:
                    if "cam0" in latest_frames and "cam1" in latest_frames:
                        f0 = latest_frames["cam0"].copy()
                        f1 = latest_frames["cam1"].copy()
                        
                        path0 = os.path.join(dir_cam0, f"cal_{capture_count:04d}.jpg")
                        path1 = os.path.join(dir_cam1, f"cal_{capture_count:04d}.jpg")
                        
                        cv2.imwrite(path0, f0)
                        cv2.imwrite(path1, f1)
                        print(f"Captured pair {capture_count:04d} -> Saved {path0} and {path1}")
                        capture_count += 1
            
            # Press 'q' to quit
            elif key == ord('q'):
                break
                
    finally:
        stop_threads = True
        t0.join()
        t1.join()
        cv2.destroyAllWindows()
        print("Done capturing calibration frames!")

if __name__ == "__main__":
    main()