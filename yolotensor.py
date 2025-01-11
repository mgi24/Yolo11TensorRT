#USING MODEL IS THE SAME AS NORMAL YOLO
from ultralytics import YOLO
from mss import mss
import cv2 as cv
import numpy as np
import time
model = YOLO("T4/yolo11m.engine", task="detect")


cropx = 416
cropy = 288
gamex=448
gamey=448
monitor  = {"top":cropy,"left":cropx, "width":gamex, "height":gamey, "monitor":0}
sct = mss()
ms=[]
fpsarr=[]
framecount=0
while True:
    image = sct.grab(monitor)
    image = np.array(image)
    frame_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    starttime = time.time() 
    results = model(frame_bgr, verbose=False, device = 0)
    elapsed_time = time.time() - starttime
    ms.append(elapsed_time*1000)
    fps = 1 / elapsed_time
    fpsarr.append(fps)
    framecount+=1
    if framecount>10:
        print(f"Average time: {np.mean(ms):.2f} ms, Average FPS: {np.mean(fpsarr):.2f}")
        ms=[]
        fpsarr=[]
        framecount=0