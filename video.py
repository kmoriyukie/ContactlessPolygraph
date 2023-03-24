import cv2

class video():
    def __init__(self, path = ""):
        self.cap = cv2.VideoCapture(path)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.len = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    def disp(self):
        if(self.cap.isOpened == False):
            print("Error opening video!")
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                cv2.imshow('Frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
        self.cap.release()
        cv2.destroyAllWindows()
            
vid = video("./data/baby.mp4")
vid.disp()