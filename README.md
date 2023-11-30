# Pose Detection with OpenCV and MediaPipe

## 1. Importing Libraries

```python
import cv2
import mediapipe as mp
import time
```

- `cv2`: OpenCV library for computer vision tasks.
- `mediapipe`: MediaPipe library for pose detection.

## 2. `PoseDetector` Class

```python
class PoseDetector():
    def __init__(self, staticImageMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=False, smoothSegmentation=True, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        # Initialization with various configuration parameters
        # ...

    def findPose(self, img, draw=True):
        # Process image and draw landmarks if detected
        # ...

    def findPosition(self, img, draw=False):
        # Find pose landmarks and draw circles on the image
        # ...
```

- The `PoseDetector` class is initialized with various parameters related to pose detection.
- The `findPose` method processes the input image and draws landmarks if detected.
- The `findPosition` method extracts and returns the positions of landmarks.

## 3. Main Function

```python
def main():
    # Open video capture
    cap = cv2.VideoCapture('Videos/V2.mp4')
    
    # Create PoseDetector instance
    detector = PoseDetector()

    # Initialize variables for FPS calculation
    pTime = 0
    cTime = 0

    # Set target FPS for video display
    target_fps = 100
    wait_key_delay = int(1000 / target_fps)

    # Main loop for video processing
    while True:
        # Read a frame from the video capture
        success, img = cap.read()

        # Detect and draw pose on the frame
        img = detector.findPose(img)
        
        # Find and print pose landmarks
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Display the frame with the drawn pose
        cv2.imshow("Image", img)

        # Break the loop on 'q' key press or window close
        if cv2.waitKey(wait_key_delay) & 0xFF == ord('q') or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Entry point of the script
if __name__ == '__main__':
    main()
```

- The `main` function sets up the video capture, creates an instance of the `PoseDetector` class, and then enters a loop to process each frame.
- Inside the loop, it detects the pose, finds pose landmarks, prints them, calculates and displays FPS, and shows the processed frame.
- The loop continues until the 'q' key is pressed or the window is closed.
- Finally, it releases the video capture and closes all windows.

### Note:

- Ensure that you have the required video file ('Videos/V2.mp4') in the specified location.
- The script uses the `mediapipe` library for pose detection, so make sure it is installed (`pip install mediapipe`).
```
