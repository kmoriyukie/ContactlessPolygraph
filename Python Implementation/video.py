import cv2
import numpy as np
class video():
    frames = []
    def __init__(self, 
                 path = "", 
                 stack = None):
        self.path = path
        if stack != None:
            self.width = len(stack[1])
            self.height = len(stack[2])
            self.fps = 30
            self.frames = np.asarray(stack)
            self.len = len(self.frames)
            
            self.displayInfo()
        else:
            cap = cv2.VideoCapture(self.path)
            if(not cap.isOpened()): 
                print("Video not found!")
                return
            self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.getFrames()
    def displayInfo(self):
        print("Length: ",self.len)
        print("Width: ",self.width)
        print("Height: ",self.height)
        print("# Frames: ",len(self.frames))
    def export(self, new_file_name=''):
        if(new_file_name==''):
            new_file_name=self.path

        height, width = self.frames[0].shape[:2]
        result = cv2.VideoWriter('./results/' + new_file_name, 
                                cv2.VideoWriter_fourcc(*'DIVX'), 
                                self.fps,
                                (width, height))
        
        for f in self.frames:
            result.write(f.astype('uint8'))            

        result.release()
    def disp(self):
        cap = cv2.VideoCapture(self.path)
        if(cap.isOpened == False):
            print("Error opening video!")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
        cap.release()
        cv2.destroyAllWindows()
    def getFrames(self):
        cap = cv2.VideoCapture(self.path)
        count = 0
        self.frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            count +=1
            if ret == True:
                self.frames.append(frame)
                if count == self.len:
                    break
            else: 
                break
        cap.release()

