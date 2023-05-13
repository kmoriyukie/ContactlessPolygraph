import numpy as np
import scipy
import cv2
import lib
from numpy import matlib as ml
def getFilter(name):
    if name == 'binom5':
        kern = [0.5, 0.5]
        for _ in range(5-2):
            kern = scipy.signal.convolve([0.5,0.5], kern)
        return np.sqrt(2) * kern
    elif name =='haar':
        return np.array([1/sqrt(2),1/sqrt(2)])
    elif name == 'gauss':
        return np.array(sqrt(2) *[0.0625, 0.25, 0.375, 0.25, 0.0625])
    
# correlationDownsample()
#  Calculates the cross correlation between the image and the filter.
#  filter: which filter to filter with
#  step: how much to downsample
#  window_start and window_stop: determine the window in which the image will be filtered
def correlationDownsample(image, filter, step = [2,2], window_stop = (-1,-1), window_start = (0,0),axis=0):
    if(window_stop == (-1,-1)):
        window_stop = (image.shape[0], image.shape[1])

    if len(filter.shape) == 1:
        # filter = filter[len(filter)-1:0:-1]
        if(axis==1):
            image = scipy.ndimage.correlate1d(image*1.0, filter,axis=0)
        else:
            image = scipy.ndimage.correlate1d(image*1.0, filter,axis=1)
    else:
        filter = filter[len(filter)-1:0:-1, len(filter)-1:0:-1]
        image = scipy.ndimage.convolve(1.0*image, filter,mode='reflect')

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
    if(isinstance(filter, str)):
        filt = getFilter(filter)
    else:
        filt = filter
    filt = filt/filt.sum()

    if levels > 1:
        image = blurDownsample(image, levels - 1, filt)

    
    if levels >= 1:
        if(image.shape[0] == 1 or image.shape[1] == 1):
            return correlationDownsample(image, filt, ((image.shape[0]!=1) * 1+1,(image.shape[1]!=1) * 1 +1))
        elif(len(filt.shape)==1):
            return correlationDownsample(correlationDownsample(image, filt, [2,1],axis=1), filt, [1,2])
        else: 
            return correlationDownsample(image, filt, [2,2])
    else:
        return image

def ConvUpsample(image, filter, step, start, stop):

    return

# idealBandPassing:
# Applies ideal bandpass filter on input
# wLow: lower cutoff region
# wUpper: upper cutoff region

def idealBandPassing(input, wLow, wUpper, samplingRate):
    dim = 1

    f = np.roll(input,dim-1)
    input = np.asarray(f)

    dimensions = list(f.shape)

    n = dimensions[0]
    print(n)
    dn = len(dimensions)

    freq = (np.linspace(1, n, n) - 1)/n*samplingRate

    mask = np.asarray((freq > wLow) & (freq < wUpper))[np.newaxis][np.newaxis][np.newaxis].transpose()

    dimensions[0] = 1
    
    mask = np.tile(mask, dimensions)
    
    f = scipy.fft.fft(f,axis=0)


    f[~mask] = 0 
    
    out = np.real(scipy.fft.ifft(f,axis=0))

    return out

# Applies butterworth filter on input
def butterFilter():


    return
# blurDownsampleStack():
# def idealBandPassing(input, wLow, wUpper, samplingRate):
#     stackOut = []

#     for s in input:
#         # print(np.asarray(idealBandPassingSingle(s, wLow, wUpper, samplingRate)).shape)
#         stackOut.append(np.asarray(idealBandPassingSingle(s, wLow, wUpper, samplingRate)))
    
#     # aux = np.asarray(input).shape
#     # shap = [a for a in aux]
#     # shap.append(3)
#     return stackOut
# A = np.array([[2, 2, 0, 6, 4, 9, 0, 1, 5, 3, 9, 3, 1, 2, 1, 8],
#               [2, 2, 0, 6, 4, 9, 0, 1, 5, 3, 9, 3, 1, 2, 1, 8],
#               [9, 2, 2, 6, 9, 1, 1, 7, 8, 0, 9, 8, 5, 6, 2, 6],
#               [4, 0, 3, 6, 6, 6, 6, 6, 3, 1, 5, 7, 1, 0, 6, 8],
#               [4, 1, 1, 4, 3, 0, 5, 0, 5, 1, 9, 6, 6, 7, 0, 9],
#               [2, 9, 5, 3, 8, 1, 7, 7, 3, 5, 1, 0, 2, 5, 5, 9],
#               [9, 7, 8, 9, 7, 1, 5, 4, 2, 7, 9, 0, 2, 7, 3, 8],
#               [1, 5, 6, 2, 8, 5, 5, 4, 7, 8, 6, 1, 4, 4, 9, 3],
#               [6, 0, 0, 5, 3, 0, 3, 1, 2, 4, 6, 4, 6, 3, 4, 8],
#               [4, 9, 8, 0, 9, 8, 3, 2, 9, 4, 5, 6, 8, 9, 9, 7],
#               [8, 1, 2, 6, 7, 3, 7, 7, 1, 7, 8, 0, 6, 6, 2, 7],
#               [6, 0, 1, 4, 0, 1, 8, 3, 8, 3, 1, 4, 2, 7, 2, 5],
#               [2, 0, 1, 3, 0, 2, 1, 5, 8, 8, 6, 9, 0, 1, 7, 6],
#               [1, 8, 5, 5, 8, 4, 6, 8, 5, 6, 4, 0, 6, 2, 8, 9],
#               [9, 0, 7, 1, 0, 6, 8, 8, 5, 4, 0, 7, 9, 8, 5, 5],
#               [2, 5, 7, 8, 9, 4, 7, 6, 4, 9, 4, 0, 2, 9, 1, 6],
#               [0, 6, 9, 1, 6, 8, 6, 0, 7, 7, 7, 0, 4, 2, 6, 9]])
# print(A.shape)
# # a = idealBandPassingSingle(A, 20.0/60, 40.0/60, 1)
# # print(a)
