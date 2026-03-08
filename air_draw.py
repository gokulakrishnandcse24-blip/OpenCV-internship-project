import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks.python import vision
MODEL_PATH="hand_landmarker.task"
CAMERA_INDEX=1
window_name="Neon Air Writing"
selected_color_name="Pink"
selected_color=(255,0,255)
canvas=None
prev_point=None
smoothed_point=None
miss_count=0
MAX_MISSES=8
ERASER_RADIUS=28
COLOR_PALETTE=[("Pink",(255,0,255)),("Cyan",(255,255,0)),("Green",(0,255,0)),("Red",(0,0,255)),("Blue",(255,0,0)),("Yellow",(0,255,255))]
color_boxes=[]
BaseOptions=mp.tasks.BaseOptions
HandLandmarker=vision.HandLandmarker
HandLandmarkerOptions=vision.HandLandmarkerOptions
VisionRunningMode=vision.RunningMode
INDEX_TIP=8
INDEX_PIP=6
MIDDLE_TIP=12
MIDDLE_PIP=10
RING_TIP=16
RING_PIP=14
PINKY_TIP=20
PINKY_PIP=18
def is_finger_up_y(landmarks,tip_id,pip_id):
    return landmarks[tip_id].y<landmarks[pip_id].y
def smooth_point(curr_pt,prev_pt,alpha=0.22):
    if prev_pt is None:
        return curr_pt
    x=int(prev_pt[0]*(1-alpha)+curr_pt[0]*alpha)
    y=int(prev_pt[1]*(1-alpha)+curr_pt[1]*alpha)
    return (x,y)
def draw_neon_line(img,pt1,pt2,color):
    if pt1 is None or pt2 is None:
        return
    cv2.line(img,pt1,pt2,color,18)
    cv2.line(img,pt1,pt2,color,12)
    cv2.line(img,pt1,pt2,color,7)
    cv2.line(img,pt1,pt2,(255,255,255),2)
def draw_palette(frame):
    global color_boxes
    color_boxes=[]
    x=10
    y=10
    w=85
    h=36
    gap=10
    for name,color in COLOR_PALETTE:
        x1,y1=x,y
        x2,y2=x+w,y+h
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,-1)
        border=(255,255,255) if name==selected_color_name else (50,50,50)
        thickness=3 if name==selected_color_name else 1
        cv2.rectangle(frame,(x1,y1),(x2,y2),border,thickness)
        text_color=(0,0,0) if name in ["Cyan","Green","Yellow"] else (255,255,255)
        cv2.putText(frame,name,(x1+8,y1+24),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,1,cv2.LINE_AA)
        color_boxes.append((x1,y1,x2,y2,name,color))
        x+=w+gap
def mouse_callback(event,x,y,flags,param):
    global selected_color,selected_color_name
    if event==cv2.EVENT_LBUTTONDOWN:
        for x1,y1,x2,y2,name,color in color_boxes:
            if x1<=x<=x2 and y1<=y<=y2:
                selected_color=color
                selected_color_name=name
                break
def cleanup(cap,landmarker):
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except:
        pass
    try:
        if landmarker is not None:
            landmarker.close()
    except:
        pass
    cv2.destroyAllWindows()
cap=cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open camera. Try CAMERA_INDEX = 0, 1, 2, or 3.")
    raise SystemExit
ret,test_frame=cap.read()
if not ret:
    cleanup(cap,None)
    print("Camera opened but no frames received.")
    raise SystemExit
options=HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),running_mode=VisionRunningMode.VIDEO,num_hands=1,min_hand_detection_confidence=0.65,min_hand_presence_confidence=0.65,min_tracking_confidence=0.65)
landmarker=HandLandmarker.create_from_options(options)
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name,mouse_callback)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break
    frame=cv2.flip(frame,1)
    if canvas is None:
        canvas=np.zeros_like(frame)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    timestamp_ms=int(time.time()*1000)
    result=landmarker.detect_for_video(mp_image,timestamp_ms)
    mode_text="SHOW HAND"
    h,w,_=frame.shape
    hand_found=False
    drawing_now=False
    if result.hand_landmarks:
        hand=result.hand_landmarks[0]
        hand_found=True
        pts=[]
        for lm in hand:
            x=int(lm.x*w)
            y=int(lm.y*h)
            pts.append((x,y))
        connections=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
        for a,b in connections:
            cv2.line(frame,pts[a],pts[b],(90,220,90),1)
        for p in pts:
            cv2.circle(frame,p,3,(0,255,0),-1)
        index_up=is_finger_up_y(hand,INDEX_TIP,INDEX_PIP)
        middle_up=is_finger_up_y(hand,MIDDLE_TIP,MIDDLE_PIP)
        ring_up=is_finger_up_y(hand,RING_TIP,RING_PIP)
        pinky_up=is_finger_up_y(hand,PINKY_TIP,PINKY_PIP)
        index_point=pts[INDEX_TIP]
        middle_point=pts[MIDDLE_TIP]
        ring_point=pts[RING_TIP]
        if index_up and not middle_up and not ring_up and not pinky_up:
            mode_text="DRAW MODE"
            drawing_now=True
            miss_count=0
            smoothed_point=smooth_point(index_point,smoothed_point,alpha=0.22)
            cv2.circle(frame,smoothed_point,10,selected_color,-1)
            if prev_point is not None:
                draw_neon_line(canvas,prev_point,smoothed_point,selected_color)
            prev_point=smoothed_point
        elif index_up and middle_up and not ring_up and not pinky_up:
            mode_text="MOVE MODE"
            miss_count=0
            cv2.circle(frame,index_point,10,(0,255,255),-1)
            prev_point=None
            smoothed_point=None
        elif index_up and middle_up and ring_up and not pinky_up:
            mode_text="ERASE MODE"
            miss_count=0
            ex=int((index_point[0]+middle_point[0]+ring_point[0])/3)
            ey=int((index_point[1]+middle_point[1]+ring_point[1])/3)
            erase_point=(ex,ey)
            cv2.circle(frame,erase_point,ERASER_RADIUS,(200,200,200),2)
            cv2.circle(canvas,erase_point,ERASER_RADIUS,(0,0,0),-1)
            prev_point=None
            smoothed_point=None
        else:
            mode_text="WAIT"
            miss_count+=1
    if not drawing_now:
        if not hand_found:
            mode_text="NO HAND"
        if miss_count>MAX_MISSES:
            prev_point=None
            smoothed_point=None
    glow=cv2.GaussianBlur(canvas,(0,0),10)
    output=cv2.addWeighted(frame,1.0,glow,0.7,0)
    output=cv2.addWeighted(output,1.0,canvas,1.0,0)
    draw_palette(output)
    cv2.putText(output,f"Color: {selected_color_name}",(10,65),cv2.FONT_HERSHEY_SIMPLEX,0.7,selected_color,2)
    cv2.putText(output,"Index = Draw",(10,95),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
    cv2.putText(output,"Index + Middle = Move",(10,122),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
    cv2.putText(output,"Index + Middle + Ring = Erase",(10,149),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2)
    cv2.putText(output,"C = Clear | S = Save | Q / ESC = Quit",(10,176),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    cv2.putText(output,f"Mode: {mode_text}",(10,205),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
    cv2.imshow(window_name,output)
    if cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE)<1:
        break
    key=cv2.waitKey(1)&0xFF
    if key==ord('c'):
        canvas=np.zeros_like(frame)
        prev_point=None
        smoothed_point=None
        miss_count=0
    elif key==ord('s'):
        filename=f"air_writing_{int(time.time())}.png"
        cv2.imwrite(filename,output)
        print(f"Saved: {filename}")
    elif key==ord('q') or key==27:
        break
cleanup(cap,landmarker)
print("Camera released. Program closed.")