import cv2
import numpy as np


def showImage(imagePath, windowName='img'):
    img = cv2.imread(imagePath)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resize(img, (int(img.shape[0]/4), int(img.shape[1]/4)))
    cv2.imshow(windowName,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ntsc2rgb(img):
    img[0] = 0.229 * img[0].astype(float)
    img[1] = 0.229 * img[1].astype(float)
    img[2] = 0.0722 * img[2].astype(float)
    return img

def rgb2ntsc(img):
    img2 = img.astype(float)
    R = img[0]
    G = img[1]
    B = img[2]
    img2[0] = 0.299 * R + 0.587 * G + 0.114 * B
    img2[1] = 0.596 * R - 0.274 * G - 0.322 * B
    img2[2] = 0.211 * R - 0.523 * G + 0.312 * B
    
    return img2
    
def normalizedImage(img):
    img = 255.0 * (img - img.min())/(img.max() - img.min())
    
    return img 