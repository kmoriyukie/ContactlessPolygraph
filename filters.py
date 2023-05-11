import numpy as np
import scipy
import cv2
import lib
from numpy import matlib as ml
def getFilter(name):
    if name == 'binom5':
        # a = np.array([1,1])
        # b = np.array([1])
        # for _ in range(5):
        #     b = np.convolve(a, b)
        # return 1/25 * np.multiply(b, b.reshape([b.shape[0],1]))
        return 1/25 * np.array([[1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1]])
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

    # image1 = image[:,:,0].squeeze()
    # image2 = image[:,:,1].squeeze()
    # image3 = image[:,:,2].squeeze()
    
    # filter= filter[:,:]

    # image[:,:,0] = scipy.ndimage.correlate(1.0*image1, filter).squeeze()
    # image[:,:,1] = scipy.ndimage.correlate(1.0*image2, filter).squeeze()
    # image[:,:,2] = scipy.ndimage.correlate(1.0*image3, filter).squeeze()
    image=np.pad(image,4,'reflect')
    filter = filter[len(filter)-1:1:-1, len(filter)-1:1:-1]
    image = scipy.signal.convolve2d(1.0*image, filter,mode='valid')

    return image[window_start[0]:window_stop[0]:step[0], window_start[1]:window_stop[1]:step[1]]
    
# blurDownsample()
# Recursively blurs and downsamples the image levels times.
# The downsampling is always done by 2 in each direction.

def blurDownsample_(image, levels,filter='binom5'):
    tmp = blurDownsample(image[:,:,1], levels)
    out = np.zeros((tmp.shape[0],tmp.shape[1],image.shape[2]))
    out[:,:,0] = tmp
    out[:,:,1] = blurDownsample(image[:,:,1], levels)
    out[:,:,2] = blurDownsample(image[:,:,2], levels)
        
    return out

def blurDownsample(image, levels, filter = 'binom5'):
    filt = getFilter(filter)
    filt = filt
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
# Applies ideal bandpass filter on input
# wLow: lower cutoff region
# wUpper: upper cutoff region

# blurDownsampleStack():
def idealBandPassing(input, wLow, wUpper, samplingRate):
    stackOut = []

    for s in input:
        # print(np.asarray(idealBandPassingSingle(s, wLow, wUpper, samplingRate)).shape)
        stackOut.append(np.asarray(idealBandPassingSingle(s, wLow, wUpper, samplingRate)))
    
    # aux = np.asarray(input).shape
    # shap = [a for a in aux]
    # shap.append(3)
    return stackOut

def idealBandPassingSingle(input, wLow, wUpper, samplingRate):
    # dim = 1
    # input = np.moveaxis(input,0, 0)
    # Transform into frequency domain
    # input = lib.rgb2ntsc(input)
    
    n = input.shape[0]
    # Get frequency of each t
    freq = np.linspace(0, n-1, n)/n*samplingRate
    #filtering
    mask = np.nonzero((freq > wLow) & (freq < wUpper))
    mask = ml.repmat(mask, 1, input.shape[1])

    f = scipy.fft.fft(input,axis=0)
    f[~mask] = 0

    out = scipy.fft.ifft(f,axis=0)

    # out = np.moveaxis(out,0, 1)
    return np.real(out).squeeze()

    
def butterFilter():
    return

