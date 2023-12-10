import cv2
import os
from pathlib import Path
ROOT = Path(__file__).parent
VIDEOS = ROOT/'videos'
IMAGES = ROOT/'images'
img_count = 0
ret, frame = None, None
for video_file in os.listdir(VIDEOS):
    cam = cv2.VideoCapture(str(VIDEOS/video_file))
    while (True):
        for i in range(10):  # skip frames
            ret, frame = cam.read()
        if ret:
            img_name = f"{img_count:04}.png"
            cv2.imwrite(str(IMAGES/img_name), frame)
            img_count += 1
        else:
            break
