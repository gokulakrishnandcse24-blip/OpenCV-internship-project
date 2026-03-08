import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
HAND_CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
MODEL_PATH="hand_landmarker.task"
def draw_hand_landmarks(frame,hand_landmarks_list):
    h,w,_=frame.shape
    for hand_landmarks in hand_landmarks_list:
        points=[]
        for landmark in hand_landmarks:
            x=int(landmark.x*w)
            y=int(landmark.y*h)
            points.append((x,y))
            cv2.circle(frame,(x,y),5,(0,255,0),-1)
        for start_idx,end_idx in HAND_CONNECTIONS:
            cv2.line(frame,points[start_idx],points[end_idx],(255,0,0),2)
def main():
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH)
    options=vision.HandLandmarkerOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=0.5,min_hand_presence_confidence=0.5,min_tracking_confidence=0.5)
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera. Try changing VideoCapture(0) to 1 or 2.")
        return
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success,frame=cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break
            frame=cv2.flip(frame,1)
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
            timestamp_ms=int(time.time()*1000)
            result=landmarker.detect_for_video(mp_image,timestamp_ms)
            if result.hand_landmarks:
                draw_hand_landmarks(frame,result.hand_landmarks)
            cv2.imshow("Hand Landmarker - Current MediaPipe",frame)
            key=cv2.waitKey(1)&0xFF
            if key==27:
                break
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()