import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

ret, image = cap.read()

if not ret:
    raise RuntimeError("failed to read frame")

image = image[:, :, [2, 1, 0]]
print(image)
im = Image.fromarray(image)
im.save('test.png')
