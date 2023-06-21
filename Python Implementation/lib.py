import cv2
import numpy as np
from collections import deque

def showImage(imagePath, windowName='img'):
    img = cv2.imread(imagePath)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resize(img, (int(img.shape[0]/4), int(img.shape[1]/4)))
    cv2.imshow(windowName,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ntsc2rgb(img):
    Y = img[:,:,0] * 1.0
    I = img[:,:,1] * 1.0
    Q = img[:,:,2] * 1.0

    img[:,:,2] = Y + 0.956 * I + 0.621 * Q #R
    img[:,:,1] = Y - 0.272 * I - 0.647 * Q #G
    img[:,:,0] = Y - 1.106 * I + 1.703 * Q #B
    return img

def rgb2ntsc(img):
    img2 = img.astype(float)
    
    R = img[:,:,2]/255.0
    G = img[:,:,1]/255.0
    B = img[:,:,0]/255.0
    
    img2[:,:,0] = 0.299 * R + 0.587 * G + 0.114 * B # Y
    img2[:,:,1] = 0.596 * R - 0.274 * G - 0.322 * B # I
    img2[:,:,2] = 0.211 * R - 0.523 * G + 0.312 * B # Q
    
    return img2
    
def normalizedImage(img):
    img = 255.0 * (img - img.min())/(img.max() - img.min())
    
    return img 

def shiftdim(array, n=None):
    if n is not None:
        if n >= 0:
            axes = tuple(range(len(array.shape)))
            new_axes = deque(axes)
            new_axes.rotate(n)
            return np.moveaxis(array, axes, tuple(new_axes))
        return np.expand_dims(array, axis=tuple(range(-n)))
    else:
        idx = 0
        for dim in array.shape:
            if dim == 1:
                idx += 1
            else:
                break
        axes = tuple(range(idx))
        # Note that this returns a tuple of 2 results
        return np.squeeze(array, axis=axes), len(axes)