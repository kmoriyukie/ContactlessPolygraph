import cv2

def showImage(imagePath, windowName='img'):
    img = cv2.imread(imagePath)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resize(img, (int(img.shape[0]/4), int(img.shape[1]/4)))
    cv2.imshow(windowName,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
