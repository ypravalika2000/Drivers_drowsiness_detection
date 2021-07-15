#importing libraries
from imutils import face_utils
import dlib
import cv2
import time
import math
from sklearn import tree
import pandas as pd
import numpy as np
import systemcheck
from playsound import playsound



chances = 0
drowsy = 0
siren = 0

endfps= 0
startfps=0
#training function
def training():
    a = pd.read_csv("blinkFatigue.csv")

    features = np.array(a['BPM']).reshape((len(a['BPM']),-1))
    labels = a['FATIGUE']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    return clf


def Euclidean_Distance(x,y):
        dis = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
        return dis
# ratio of eye lid distance 
def eye_aspect_ratio(eye):
        A = Euclidean_Distance(eye[1], eye[5])
        B = Euclidean_Distance(eye[2], eye[4])
        C = Euclidean_Distance(eye[0], eye[3])
        D = (A+B)**2
        ear = D / (2.0 * C)
        return ear

def mouth_aspect_ratio(mouth):
        A = Euclidean_Distance(mouth[0], mouth[6])
        A = A**2
        B = Euclidean_Distance(mouth[3], mouth[9])
        B = B**2

        ear = A/B
        return ear

blink = 0
yawn = 0
lastBlink = 0
blinkDur = 0
op = 0
timer1 = 0
thresh = 3.5
frame_check = 5
detect = dlib.get_frontal_face_detector()
# detect = dlib.cnn_face_detection_model_v1("human_face_detector.dat")
# heart of the code
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(minStart, minEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["inner_mouth"]
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

#print(face_utils.FACIAL_LANDMARKS_68_IDXS)
# video capturing is started

cap=cv2.VideoCapture(0)
flag=0
flag1 = 0
count = 0
start = time.time()

clf = training()

while True:

        siren = 0
        ret, frame= cap.read()
        count += 1
        #frame = imutils.resize(frame, width=450)
        # each frame is converted to gray color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        try:
                subject = list(subjects)[0]
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)#converting to NumPy Array
                #print(len(shape))
                leftEye = shape[lStart:lEnd]
                leftEAR = eye_aspect_ratio(leftEye)

                rightEye = shape[rStart:rEnd]
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                #print('EAR :',ear)


                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                mouth = shape[mStart:mEnd]
                mouthHull = cv2.convexHull(mouth)

                mouthEAR = mouth_aspect_ratio(mouth)

                #print("Mouth Ratio" , mouthEAR)

                nose = shape[nStart:nEnd]
                noseHull = cv2.convexHull(nose)

                # jaw = shape[jStart:jEnd]
                # jawHull = cv2.convexHull(jaw)

                re = shape[reStart:reEnd]
                reHull = cv2.convexHull(re)

                le = shape[leStart:leEnd]
                leHull = cv2.convexHull(le)

                cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (255, 255, 0), 1)
                #cv2.drawContours(frame, [noseHull], -1, (255, 255, 0), 1)
                #cv2.drawContours(frame, [jawHull], -1, (255, 255, 0), 1)
                cv2.drawContours(frame, [reHull], -1, (255, 255, 0), 1)
                cv2.drawContours(frame, [leHull], -1, (255, 255, 0), 1)
                # print('EAR: ', ear)
                if ear < thresh:

                        if flag == 0 and time.time()-lastBlink > 1:
                                blink += 1
                                lastBlink = time.time()
                                print("Blink Detected", blink)
                                2

                        print (flag,end=' ')
                        flag += 1
                        if(flag > 10):
                                cv2.putText(frame, "  STAY ALERT ", (200, 250),   # warning display
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                                cv2.putText(frame, " DON'T SLEEP ", (200,400),    # warning display
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                                siren = 1


                else:
                        flag = 0

                if(time.time()- start > 30):
                        print("Blink Per minute :",blink)     #BPM
                        p = np.array([blink]).reshape((1,-1))
                        op = clf.predict(p)
                        print('Chances of Drowsy :', op[0])
                        start = time.time()
                        blink = 0
                        timer1 = 0

                if op[0] > 0:
                        #print("Drowsy")
                        cv2.putText(frame, "  STAY ALERT ", (200, 250),            # warning display
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                        cv2.putText(frame, " YOU MAYBE SLEEPY ", (200,400),        # warning display
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                        drowsy = 1
                        siren = 1
                        if(timer1 == 0):
                                start= time.time()
                                timer1 = 1 

                        elif(timer1 == 1 and ((time.time()- start) > 10)):
                                op=0


                if(mouthEAR < 5):
                        flag1 += 1
                        if flag1 > frame_check:
                                yawn += 1
                                print("Yawn detected")      # warning display
                                flag1 = 0

                else:
                        flag1 = 0
        except:
                pass



        endfps = time.time()
        fps = int(1/(endfps-startfps))
        cv2.putText(frame, "FPS:- "+str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        if(time.time() - start > 60):
                print('Yawns ', yawn)
                print('BPM : ', blink)

                if(yawn > 1 or (3 < blink < 6)):
                        cv2.putText(frame, "Chances of Drowsiness Soon", (1, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                        chances = 1
                        siren = 1

                yawn = 0
                blink = 0
                start = time.time()

        startfps = time.time()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)     #press esc to close the window
        if key == 27:
                break
        if siren == 1:
            playsound("siren.wav")    #soundfile
cv2.destroyAllWindows()
cap.release()
