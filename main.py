import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, staticImageMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=False, smoothSegmentation=True, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.staticImageMode = staticImageMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.staticImageMode, self.modelComplexity, self.smoothLandmarks, self.enableSegmentation, self.smoothSegmentation, self.minDetectionConfidence, self.minTrackingConfidence)
    
    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPosition(self, img, draw=False):
        lmList = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            lmList.append([id, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return lmList

    

def main():
    cap = cv2.VideoCapture('Videos/V2.mp4')
    detector = PoseDetector()

    pTime = 0
    cTime = 0

    target_fps = 100
    wait_key_delay = int(1000 / target_fps)

    while True:
        success, img = cap.read()

        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(wait_key_delay) & 0xFF == ord('q') or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

        # time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()