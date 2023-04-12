import filters
import video as vid
import numpy as np


# buildGaussianStack()
# Build gaussian pyramid with level layers
# video: video class object
# temporalWindow: How big of a timeframe (in video) are we taking 
# in consideration?
# level: how many layers in our laplacian decomposition pyramid?
def buildGaussianStack(video, temporalWindow, level):
    stack = []
    for i in range(temporalWindow[0], temporalWindow[1]):
        # print(i)
        stack.append(filters.blurDownsample(video.frames[i], level))
    return stack
    # cv2.destroyAllWindows()

