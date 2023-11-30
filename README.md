# Pose Detection with OpenCV and MediaPipe

## Overview

This Python script utilizes OpenCV and MediaPipe to perform real-time pose detection on a video stream. The script captures video frames, detects human poses, and displays the results with key landmarks.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe

Install the required libraries using:

```bash
pip install opencv-python
pip install mediapipe
```

## Usage

1. Clone the repository or download the script.

```bash
git clone https://github.com/devDurgeshK/poseEstimationModule.git
cd poseEstimationModule
```

2. Run the script:

```bash
python main.py
```

3. Press 'q' to exit the program.

## Configuration

The `PoseDetector` class can be configured by adjusting the initialization parameters in the script:

- `staticImageMode`: Set to `True` for static image mode.
- `modelComplexity`: Model complexity parameter (default: 1).
- `smoothLandmarks`: Enable smoothing of landmark points.
- `enableSegmentation`: Enable segmentation (not used in this script).
- `smoothSegmentation`: Enable smoothing of segmentation masks.
- `minDetectionConfidence`: Minimum confidence threshold for detection (default: 0.5).
- `minTrackingConfidence`: Minimum confidence threshold for tracking (default: 0.5).

### Functionality:

1. **Initialization:**
    
    - The class is initialized with various configuration parameters related to pose detection.
2. **MediaPipe Setup:**
    
    - It initializes instances of the `drawing_utils` and `pose` modules from the `mediapipe` library.
3. **`findPose` Method:**
    
    - This method takes an image (`img`) as input and detects the pose using the MediaPipe Pose model.
    - If the `draw` parameter is True, it draws landmarks on the image.
    - Returns the modified image.
4. **`findPosition` Method:**
    
    - This method takes an image (`img`) as input and extracts the positions of pose landmarks.
    - If the `draw` parameter is True, it draws circles around the landmarks on the image.
    - Returns a list (`lmList`) containing landmark positions [id, x, y].

### Note:

- The `PoseDetector` class encapsulates the functionality related to pose detection using the MediaPipe library.
- It provides methods for detecting poses in images and extracting pose landmark positions.
- The class is designed to be reusable and configurable with various parameters.

## Contributing

Feel free to contribute to the project by submitting issues or pull requests. Your feedback and enhancements are welcomed!

Below is the functionality of the `findPose` and `findPosition` methods in the `PoseDetector` class:

### `findPose` Method:

```python
def findPose(self, img, draw=True):
    """
    Detects the pose in the given image and optionally draws the pose landmarks.

    Parameters:
    - img: The input image.
    - draw: If True, draw the pose landmarks on the image.

    Returns:
    - img: The image with or without drawn landmarks.
    """
    # Convert image to RGB format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image with MediaPipe Pose model
    self.results = self.pose.process(imgRGB)
    
    # Draw landmarks on the image if detected
    if self.results.pose_landmarks and draw:
        self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

    return img
```

#### Functionality:

1. **Image Conversion:**
   - Converts the input image (`img`) from BGR format to RGB format, as required by the MediaPipe Pose model.

2. **Pose Detection:**
   - Processes the RGB image using the MediaPipe Pose model (`self.pose.process()`).
   - The results are stored in the `self.results` attribute.

3. **Drawing Landmarks:**
   - If the pose landmarks are detected (`self.results.pose_landmarks`) and the `draw` parameter is True, it draws the landmarks on the image using the `draw_landmarks` method from `mediapipe`.

4. **Return:**
   - Returns the modified image (`img`) with or without drawn landmarks.

### `findPosition` Method:

```python
def findPosition(self, img, draw=False):
    """
    Finds the positions of pose landmarks in the given image and optionally draws circles around them.

    Parameters:
    - img: The input image.
    - draw: If True, draw circles around the pose landmarks.

    Returns:
    - lmList: A list containing landmark positions [id, x, y].
    """
    lmList = []
    for id, lm in enumerate(self.results.pose_landmarks.landmark):
        # Get image dimensions
        h, w, c = img.shape
        # Convert normalized landmark coordinates to pixel values
        cx, cy = int(lm.x * w), int(lm.y * h)

        # Append landmark information to the list
        lmList.append([id, cx, cy])

        # Draw a circle around the landmark if required
        if draw:
            cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

    return lmList
```

#### Functionality:

1. **Landmark Iteration:**
   - Iterates through the pose landmarks obtained from the MediaPipe results (`self.results.pose_landmarks`).

2. **Coordinate Conversion:**
   - Converts the normalized landmark coordinates to pixel values using the dimensions of the input image.

3. **Landmark List:**
   - Appends the landmark information to the `lmList` list in the format [id, x, y].

4. **Drawing Circles:**
   - If the `draw` parameter is True, it draws circles around the landmarks on the input image.

5. **Return:**
   - Returns the `lmList` containing landmark positions.

### Note:
- `findPose` is responsible for detecting the overall pose in the image and drawing landmarks if required.
- `findPosition` extracts and returns the positions of individual pose landmarks, and optionally draws circles around them on the image.
- Both methods utilize the results obtained from the MediaPipe Pose model during the pose detection process.
## Example Usage

```python
from PoseDetector import PoseDetector
import cv2

# Create PoseDetector instance
detector = PoseDetector()

# Open video capture
cap = cv2.VideoCapture('path/to/your/video.mp4')

while True:
    success, img = cap.read()

    # Detect and draw pose
    img = detector.findPose(img)
    
    # Find and print pose landmarks
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList)

    # Display frame with FPS
    cv2.imshow("Pose Detection", img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
