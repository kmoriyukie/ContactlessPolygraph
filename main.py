import filters
import lib
import cv2
import numpy as np
import scipy
import video
def main():
    img = cv.imread('./data/cute.cute.jpg')
    
    cv.imshow(img)
    # return 0

# main()

vid = video.video("./data/baby.mp4")
vid.disp()
# vid.getFrames()

print(len(vid.fraqmes))