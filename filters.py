import numpy as np
import scipy
import cv2
def getFilter(name):
    if name == 'binom5':
        a = np.array([1,1])
        b = np.array([1])
        for _ in range(5):
            b = np.convolve(a, b)
        return 1/25 * np.multiply(b, b.reshape([b.shape[0],1]))
    elif name =='haar':
        return np.array([1/sqrt(2),1/sqrt(2)])
    elif name == 'gauss':
        return np.array(sqrt(2) *[0.0625, 0.25, 0.375, 0.25, 0.0625])
    
# correlationDownsample()
#  Calculates the cross correlation between the image and the filter.
#  filter: which filter to filter with
#  step: how much to downsample
#  window_start and window_stop: determine the window in which the image will be filtered
def correlationDownsample(image, filter, step = [2,2], window_stop = (-1,-1), window_start = (0,0)):
    if(window_stop == (-1,-1)):
        window_stop = (image.shape[0], image.shape[1])

    cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image1 = image[:,:,0]
    image2 = image[:,:,1]
    image3 = image[:,:,2]

    image1 = image1[:,:,None]
    image2 = image2[:,:,None]
    image3 = image3[:,:,None]
    
    filter= filter[:,:,None]
    image[:,:,0] = scipy.ndimage.correlate(1.0*image1, filter).squeeze()
    image[:,:,1] = scipy.ndimage.correlate(1.0*image2, filter).squeeze()
    image[:,:,2] = scipy.ndimage.correlate(1.0*image3, filter).squeeze()

    return image[window_start[0]:window_stop[0]:step[1], window_start[1]:window_stop[1]:step[0],:]
    
# blurDownsample()
# Recursively blurs and downsamples the image levels times.
# The downsampling is always done by 2 in each direction.
def blurDownsample(image, levels, filter = 'binom5'):
    filt = getFilter(filter)
    filt = filt/filt.sum()
    if levels > 1:
        image = blurDownsample(image, levels - 1, filter)
    if levels >= 1:
        if(image.shape[0]==1 or image.shape[1] ==1):
            return correlationDownsample(image, filt)
        if(filt.shape == (1,)):
            return correlationDownsample(correlationDownsample(image, filt, [2,1]), filt, [1,2])
        else: 
            return correlationDownsample(image, filt)
    else:
        return image

# idealBandPassing:
# Applies bandpass filter on input
# wLow: lower cutoff region
# wUpper: upper cutoff region

def idealBandPassing(input, dim, wLow, wUpper, samplingRate):
