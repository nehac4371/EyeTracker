import cv2
import numpy as np
import dlib

class PupilTrack:
    
    """
    Track pupil of the eye using Webcam.
    """
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):

        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow('image')

    
    def detect_eye(self,eye_points,facial_landmarks):
    
        eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                        (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                        (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                        (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                        (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                        (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        return eye_region

    
    def get_pupil_rect(self,eye_region):

        lt = (eye_region[0][0]+5,eye_region[1][1]-1)
        rt = (eye_region[3][0]-5,eye_region[1][1]-1)
        rb = (eye_region[3][0]-5,eye_region[4][1]+1)
        lb = (eye_region[0][0]+5,eye_region[4][1]+1)
    
        return np.array([lt,rt,rb,lb])


    def get_connected_components(self,threshold_eye,eye):
       
        
        output = cv2.connectedComponentsWithStats(threshold_eye, 4)
            
        stats = output[2]
        centroids = output[3]
        
        #getting the index of the component with the largesr area
        area = []
        i = 0;
        for stat in stats:
            area.append((stat[4],i))
            i = i+1
        
        M = max(area)
        ind = M[1]
    
        ##getting the centroid of the largest component    
        pupcentre = centroids[ind] 
        

        x = int(pupcentre[0])
        y = int(pupcentre[1])
    
        cv2.circle(threshold_eye, (x,y),1,(0,0,0), -1)
        
        return pupcentre[0],pupcentre[1]

    def get_pupil(self,eye_region,threshold,height,width,gray):
    
        mask = np.zeros((height, width), np.uint8)
        eye_pnts = self.get_pupil_rect(eye_region)
    
        cv2.fillConvexPoly(mask,eye_pnts,4,255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
    
        #getting dimension of the eye
        min_x = np.min(eye_region[:, 0]) +10
        max_x = np.max(eye_region[:, 0]) -10
        min_y = np.min(eye_region[:, 1]) +1
        max_y = np.max(eye_region[:, 1]) -1

        #gray image of just the eye
        gray_eye = eye[min_y: max_y, min_x: max_x]
        
        blur = cv2.medianBlur(gray_eye,5)
        threshold_eye = cv2.threshold(blur,threshold,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
       
        
        if type(threshold_eye) is not type(None):
            height, width = threshold_eye.shape
            
            #cv2.imshow('th_eye',threshold_eye)
    
            x,y =self.get_connected_components(threshold_eye,eye)
        
            return (x+min_x,y+min_y)

    def nothing(self):
        pass

    def start(self):
        cv2.createTrackbar('threshold', 'image', 0, 255,self.nothing)
        
        while True:
            retval, frame = self.cap.read()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            height, width, _ = frame.shape
            faces,scores,idx = self.detector.run(gray,0,0)

            landmarks = self.predictor(gray, faces[0])

            eye_region = self.detect_eye([36, 37, 38, 39, 40, 41],landmarks)
        
            eye_pnts = self.get_pupil_rect(eye_region)
            
            cv2.rectangle(frame,(eye_pnts[0][0],eye_pnts[0][1]),(eye_pnts[2][0],eye_pnts[2][1]),(255,0,0),1)
            
            threshold = cv2.getTrackbarPos('threshold','image')
            
            pup_center = self.get_pupil(eye_region,threshold,height,width,gray)  

            cv2.circle(frame,(int(pup_center[0]),int(pup_center[1])),2,(0,0,255),-1)
        
            cv2.imshow('image', frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


#run the code
pt = PupilTrack()
pt.start()


                

