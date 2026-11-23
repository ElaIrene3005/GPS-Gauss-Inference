import threading
import cv2
import os

def capture_camera(cam_source, folder):
    os.makedirs(folder, exist_ok=True)
    
    cap = cv2.VideoCapture(cam_source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_source}")
        return

    for i in range(200):
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {cam_source} failed to read frame.")
            break
            
        filename = os.path.join(folder, f"{i:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Cam {cam_source} saved frame {i}")
        
    cap.release()

# cam 0: assigned to dev/video0 with dev/video1 as its metadata
# cam 1: assigned to dev/video2 with dev/video3 as its metadata
t0 = threading.Thread(target=capture_camera, args=("/dev/video0", "cam0"))
t1 = threading.Thread(target=capture_camera, args=("/dev/video2", "cam1"))

t0.start()
t1.start()

t0.join()
t1.join()

print("Done capturing!")