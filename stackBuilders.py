import filters
import video as vid
# buildGaussianStack:
# The goal here is to amplify the color changes in the video
# as blood flows through the face. 
# filename: video file name
# temporalWindow: How big of a timeframe (in video) are we taking 
# in consideration?
# level: how many layers in our laplacian decomposition pyramid?
def buildGaussianStack(video, temporalWindow, level):
    if(video.cap.isOpened == False):
        print("Error opening video!")
    count = 0
    GaussianStack = []
    while(video.cap.isOpened()):
        ret, frame = video.cap.read()
        count +=1
        if ret == True:
            # cv2.imshow('Frame',frame)
            blurredImage = filters.blurDownsample(frame, level)
            GaussianStack.append(blurredImage)
            if count == video.len:
                break
        else: 
            break
    video.cap.release()
    # cv2.destroyAllWindows()