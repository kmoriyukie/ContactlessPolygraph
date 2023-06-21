import filters
import video as vid
import numpy as np
import lib
import laplacian
# buildGaussianStack()
# Build gaussian pyramid with level layers
# video: video class object
# temporalWindow: How big of a timeframe (in video) are we taking 
# in consideration?
# level: how many layers in our laplacian decomposition pyramid?
def buildGaussianStack(video, temporalWindow, level):
    stack = []
    for i in range(temporalWindow[0], temporalWindow[1]):
        image = lib.rgb2ntsc(video.frames[i])
        stack.append(filters.blurDownsample_(image, level))
    return stack
    # cv2.destroyAllWindows()

#buildLaplacianStack()
# Build Laplacian Pyramid. The second dimension is the color channel,
# the third dimension is the time.
# - video
# - window [startIndex, endIndex]
def buildLaplacianStack(video, window, height=None):
    stack = []
    stackId = []
    for i in range(window[0], window[1]):
        image = lib.rgb2ntsc(video.frames[i])
        lapl, idx = laplacian.buildLaplacian_(np.array(image))
        stack.append(lapl)
        stackId.append(idx)
    return stack, stackId
