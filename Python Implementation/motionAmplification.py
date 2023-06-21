import laplacian
import video as vid
import stackBuilders
import numpy as np
import cv2
import lib
import filters

def motionAmplification(video, alpha, lambda_c, r1, r2, chromAttenuation):
    temporalWindow = [1, int(video.len)]
    print(temporalWindow)
    image = lib.rgb2ntsc(video.frames[0])

    pyr, idx = laplacian.buildLaplacian_(image)

    lopass1 = pyr
    lopass2 = pyr

    levels = len(idx)
    print(levels)
    stack = []
    for i in range(temporalWindow[0], temporalWindow[1]):
        image = lib.rgb2ntsc(video.frames[i])

        pyr, idx = laplacian.buildLaplacian_(image)
        # print(pyr)
        lopass1 = (1.0 - r1)*lopass1 + r1 * pyr
        lopass2 = (1.0 - r2)*lopass2 + r2 * pyr
        # print('lopass1=',lopass1)
        # print('lopass2=',lopass2)
        filtered = (lopass1 - lopass2)

        ind = pyr.shape[0] -1
        
        delta = lambda_c/8.0/(1+alpha)
        exaggeration_factor = 2

        lambda_ = np.sqrt(video.height*video.height + video.width*video.width)/3
        # print(lambda_)
        for l in range(levels-1,0,-1):
            # print(idx[l])
            indices = np.array(range(ind-np.prod(idx[l]), ind-1))
            currentAlpha = lambda_/delta/8.0 - 1.0
            currentAlpha *= exaggeration_factor
            # print(filtered.shape)
            if(l == levels-1 or l == 0):
                filtered[indices,:] = 0.0
            elif(currentAlpha > alpha):
                filtered[indices,:] *= alpha
            else:
                filtered[indices,:] *= currentAlpha

            ind -= np.prod(idx[l])
            # print(ind)
            lambda_ /= 2.0
        # print(filtered)
        # return
        # filtered_ = filtered[:,:,i].squeeze()
        output = np.zeros([int(video.height), int(video.width), 3])
        output[:,:,0] = laplacian.rebuildLaplacian(filtered[:,0], np.array(idx))
        output[:,:,1] = laplacian.rebuildLaplacian(filtered[:,1], np.array(idx)) * chromAttenuation
        output[:,:,2] = laplacian.rebuildLaplacian(filtered[:,2], np.array(idx)) * chromAttenuation
        # print(output)
        output += image

        output = lib.ntsc2rgb(output)
        output = lib.normalizedImage(output)

        output[output < 0] = 0
        output[output > 255] = 255

        stack.append(output)
            # print('Level ', l, ' done!')
        # print(filtered)
        # return

    # [pyramidStack, idx] = stackBuilders.buildLaplacianStack(video, temporalWindow)

    # print(pyramidStack.shape)
    # return
    # print("Temporal filtering...")
    # filteredStack = filters.idealBandPassing(pyramidStack, fl, fh, samplingRate,dim=3)
    # print("Done!")

    #### FIND ERROR HERE!
    # levels = np.asarray(pyramidStack[:][0][0]).shape[0]
    # ######
    # stack = []
    # ####
    # exaggeration_factor = 2

    
    # delta = lambda_c/8/(1+alpha)

    # # for i in range(temporalWindow[0], temporalWindow[1]):
    #     # print(out.shape)
        
    # lambda_ = np.sqrt(video.height*video.height + video.width*video.width)/3\


    # ind = np.asarray(pyramidStack).shape[0]
    # print(filteredStack.shape)
    # return

    # for l in range(levels-1,0,-1):
    #     # print()
    #     indices = np.array(range(ind-np.prod(idx[l][:]) + 1, ind))

    #     currentAlpha = lambda_/delta/8 - 1
    #     currentAlpha *= exaggeration_factor
    #     print(filteredStack.shape)
    #     if(l == levels or l == 0):
    #         filteredStack[:,:,indices] = 0
    #     elif(currentAlpha > alpha):
    #         filteredStack[:,:,indices] *= alpha
    #     else:
    #         filteredStack[:,:,indices] *= currentAlpha

    #     ind -= np.prod(idx[l,:])
    #     lambda_ /= 2
    #     print('Level ', l, ' done!')
    
    # stack=[]
    # for i in :
        # output = np.zeros(image.shape)
        
    v = vid.video(stack=stack,path="baby2.avi")
    v.export()

        

v = vid.video(path="data/baby.mp4")
motionAmplification(v, 10, 16, 0.4, 0.05, 0.1)
