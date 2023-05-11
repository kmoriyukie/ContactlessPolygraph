import filters
import video as vid
import stackBuilders
import numpy as np
import cv2
import lib


# colorAmplification()
# Inpired by the funciton amplify_spatial_Gdown_temporal_ideal
# Applies Gaussian blur as the spatial filtering, followed by an ideal bandpass filter
# The idea is to amplify color variations, I.E. from this we will extract heart rate.
# alpha: amplification factor
# level: How many levels in our Gaussian Decomposition stack?
# bandpassRange: [wLow, wHigh] range in which our bandpass filter will operate
# chromAttenuation: attenuate color?
def colorAmplification(video, alpha, level, bandpassRange, samplingRate, chromAttenuation):
    temporalWindow = [0, int(video.len)]

    print("Spatial Filtering...")
    gaussianStack = stackBuilders.buildGaussianStack(video, temporalWindow, level)
    print("Done!")

    print("Temporal filtering...")
    filteredStack = filters.idealBandPassing(gaussianStack, bandpassRange[0], bandpassRange[1], samplingRate)
    print("Done!")

    # filteredStack[:][:][:][0] = filteredStack[:][:][:][0] * alpha
    # filteredStack[:][:][:][1] = filteredStack[:][:][:][1] * alpha * chromAttenuation
    # filteredStack[:][:][:][2] = filteredStack[:][:][:][2] * alpha * chromAttenuation
    
    
    stack = []
    
    print(len(filteredStack))
    for i in range(len(filteredStack)):
        
        filtered = filteredStack[i].squeeze()
        
        filtered[:,:,0] = filtered[:,:,0] * alpha
        filtered[:,:,1] = filtered[:,:,1] * alpha * chromAttenuation
        filtered[:,:,2] = filtered[:,:,2] * alpha * chromAttenuation

        f = cv2.resize(filtered,(int(video.width), int(video.height)))

        f = f + lib.rgb2ntsc(video.frames[i])
       
        f = lib.ntsc2rgb(f)

        f = lib.normalizedImage(f)

        f[f<0] = 0
        f[f>255] = 255
        stack.append(f)
    v = vid.video(stack=stack,path="newfil.avi")
    v.export()

def colorAmplification_Butter(video, alpha, level, bandpassRange, samplingRate, chromAttenuation):
    temporalWindow = [0, int(video.len)]

    print("Spatial Filtering...")
    gaussianStack = stackBuilders.buildGaussianStack(video, temporalWindow, level)
    print("Done!")

    print("Gaussian Stack Size: ", np.asarray(gaussianStack).shape)


    print("Temporal filtering...")
    filteredStack = filters.idealBandPassing(gaussianStack, bandpassRange[0], bandpassRange[1], samplingRate)
    print("Done!")

    print(np.asarray(filteredStack).shape)
    filteredStack[:][:][:][0] = filteredStack[:][:][:][0] * alpha
    filteredStack[:][:][:][1] = filteredStack[:][:][:][1] * alpha * chromAttenuation
    filteredStack[:][:][:][2] = filteredStack[:][:][:][2] * alpha * chromAttenuation
    
    stack = []
    
    print(len(filteredStack))
    for i in range(len(filteredStack)):
        filtered = filteredStack[i].squeeze()
        f = cv2.resize(filtered,(int(video.width), int(video.height)))
        f = lib.rgb2ntsc(f) + lib.rgb2ntsc(video.frames[i])
       
        f = lib.ntsc2rgb(f)
        f = lib.normalizedImage(f)

        stack.append(f)
    v = vid.video(stack=stack,path="newfile.mp4")
    v.export()


v = vid.video(path="data/face.mp4")
colorAmplification(v,50, 4, [50.0/60, 60.0/60], 30, 1)
