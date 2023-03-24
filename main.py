import filters
import lib
import cv2
def main():
    img = cv.imread('./data/cute.cute.jpg')
    
    cv.imshow(img)
    # return 0

# main()

# lib.showImage('./data/cute.jpg')

img = filters.blurDownsample(cv2.imread('./data/cute.jpg'), 2, 'binom5')
# print(img.dtype)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()