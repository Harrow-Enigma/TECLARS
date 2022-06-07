import os
import cv2
import time
import picamera
import numpy as np
from datetime import datetime

def timesig():
    return datetime.strftime(datetime.now(), '%Y_%m_%d_%H%M%S')

os.system('espeak "Hi guys. TECLARS system initiating."')

RESOLUTION = (512, 512)
THRESHOLD = 0.4
BOUND = False

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

SESSION = timesig()
if not os.path.exists(SESSION):
    os.mkdir(SESSION)

with picamera.PiCamera() as camera:
    camera.resolution = RESOLUTION
    #camera.framerate = 24
    time.sleep(2)
    frame = np.empty((RESOLUTION[0], RESOLUTION[1], 3), dtype=np.uint8)
    
    os.system('espeak "Hi guys. TECLARS system is ready for whatever life throws at it."')
    
    while 1:
        camera.capture(frame, 'bgr')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        print(len(faces))
        
        if len(faces) > 0:
            
            max_ratio = 0
            
            for (x,y,w,h) in faces:
                
                ratio = w / RESOLUTION[0]
                
                if ratio > THRESHOLD and BOUND:
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                if ratio > max_ratio:
                    max_ratio = ratio
            
            if max_ratio > THRESHOLD:
                cv2.imwrite(f'{SESSION}/{timesig()}.jpg', frame)
                os.system('espeak "Awesome. Face captured."')
                print("Face big enough")
            
            elif max_ratio > 0.2:
                os.system('espeak "Come closer!"')
       
        time.sleep(2)
