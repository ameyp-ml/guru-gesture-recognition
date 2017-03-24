import cv2
import numpy as np
from augment import get_uber_video

CHALAP = "/media/amey/76D076A5D0766B6F/chalap"
batches = get_batches(CHALAP, 5, 32, 401, 470, 0, 0, 0, True)
i = 0
for b in batches:
    i += 1

print(i)
