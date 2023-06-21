import numpy as np
import scipy
import cv2
import lib
from filters import getFilter, correlationDownsample

def buildLaplacian_(image, height=None,f1='binom5',f2='binom5'):
    [tmp, idx] = buildLaplacian(image[:,:,0], height, f1, f2)

    aux = np.array(np.array(idx).shape)
    aux2 = np.array(np.array(tmp).shape)
    
    aux = np.append(aux,[3])
    aux2 = np.append(aux2, [3])
    idx_ = np.zeros(aux)
    out = np.zeros(aux2)

    out[:,0] = tmp
    idx_[:,:,0] = idx
    out[:,1],idx_[:,:,1] = buildLaplacian(image[:,:,1], height, f1, f2)
    out[:,2],idx_[:,:,2] = buildLaplacian(image[:,:,2], height, f1, f2)
    
    return out, idx

def buildLaplacian(image, height=None, filt1='binom5', filt2='binom5'):
    if(isinstance(filt1, str)):
        filt1 = getFilter(filt1)
    else:
        filt1 = filt1
    if(isinstance(filt2, str)):
        filt2 = getFilter(filt2)
    else:
        filt2 = filt2

    maxHeight = 1 + maxPyramidHeight(np.array(image.shape), np.array([filt1.shape[0],filt2.shape[0]]).max())
    
    if(height==None): height = maxHeight
    
    if(height <= 1):
        pyramid = image.flatten('C')
        indices = [image.shape]
        return [pyramid, indices]
    
    if(image.shape[0] == 1):
        lo2 = correlationDownsample(image, filt1, step=[2, 1])
    elif(image.shape[1] == 1):
        lo2 = correlationDownsample(image, filt1, step=[1, 2],axis=1)
    else:
        lo = correlationDownsample(image, filt1, step=[1, 2],axis=1)
        int_size = [lo.shape[0], lo.shape[1]]
        lo2 = correlationDownsample(lo, filt1, [2, 1])
    
    [nPyramid, nIndex] = buildLaplacian(lo2, height-1)

    if(image.shape[0] == 1):
        hi2 = ConvUpsample(lo2, filt2, step=[1, 2],stop=image.shape,axis=1)
    elif(image.shape[1] == 1):
        hi2 = ConvUpsample(lo2, filt2, step=[2, 1],stop=image.shape)
    else:
        hi = ConvUpsample(lo2, filt2, step=[2, 1], stop=int_size)
        
        hi2 = ConvUpsample(hi, filt2, step=[1, 2],stop=image.shape,axis=1)
        
    hi2 = image - hi2
    hi2 = hi2.flatten('C')
    
    pyr = np.append(hi2, np.array(nPyramid))

    indices = [image.shape]
    for i in nIndex:
        indices.append(i)
    return [pyr, indices]
        

def maxPyramidHeight(imageSize, filterSize):
    if(isinstance(filterSize, int)):
        filterSize = np.array(filterSize)[np.newaxis]
    
    if(imageSize[0]==1 or imageSize[1]==1):
        imageSize = np.prod(imageSize)
        filterSize = np.prod(filterSize)
    elif (len(filterSize.shape) == 1):
        filterSize = [filterSize[0], filterSize[0]]

    if(sum(1.0*(imageSize < filterSize) ) > 0):
        height = 0
    else:
        height = 1 + maxPyramidHeight(np.floor(imageSize/2), filterSize)
    return height

def rebuildLaplacian(pyramid, indices, levels=None, filt2='binom5'):
    maxLev = indices.shape[0]

    if(levels is None): levels = np.array(range(0, maxLev))

    if(isinstance(filt2, str)):
        filt2 = getFilter(filt2)
    else:
        filt2 = filt2

    resSize = indices[0].squeeze()
    
    if(sum(1.0*(levels>1))>1):
        intSize = [indices[0][0], indices[1][1]]
        nres = rebuildLaplacian(pyramid[np.prod(resSize):pyramid.shape[0]], indices[1:indices.shape[0],:], levels-1,filt2)
        
        if(resSize[0] == 1):
            res = ConvUpsample(nres,filt2, [1,2],stop=resSize,axis=1)
        elif(resSize[1] == 1):
            res = ConvUpsample(nres,filt2, [2,1],stop=resSize)
        else:
            hi = ConvUpsample(nres, filt2, [2,1], stop=intSize)
            res = ConvUpsample(hi, filt2, [1,2],stop=resSize,axis=1)
    else:
        res = np.zeros(resSize)

    if(sum(1.0*(levels==1))>1):
        res += pyramidSubBand(pyramid, indices, 1).squeeze()
    return res

def pyramidSubBandIndices(pIndex, band):
    ind = 0

    for i in range(band-2):
        ind += np.prod(pIndex[i,:])

    return range(ind, ind+np.prod(pIndex[band,:]))


def pyramidSubBand(pyramid, pyramidIndex, band):
    return(np.reshape(pyramid[pyramidSubBandIndices(pyramidIndex, band)], [pyramidIndex[band,0], pyramidIndex[band,1]]))

def ConvUpsample(image, filt, step, start=np.array([0,0]), stop=None, res=None,axis=0):
    if(stop is None):
        stop = step * np.array(np.floor((start-np.ones(np.asarray(start).shape))/step + [1,1]) + image.shape).astype(int)
    if(res is None):
        res = np.zeros(stop-start)
    # upsample
    
    tmp = np.zeros(res.shape)
    tmp[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]]=image

    out = scipy.ndimage.convolve1d(input=1.0*tmp, weights=filt,axis=axis,mode='mirror') + res
    print(out.shape)
    return out
