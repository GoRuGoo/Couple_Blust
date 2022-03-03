# Import required modules
import cv2 as cv
import math
import time
import argparse

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


#学習済みモデルの読み込み
faceProto = "./cascade/opencv_face_detector.pbtxt"
faceModel = "./cascade/opencv_face_detector_uint8.pb"
ageProto = "./cascade/age_deploy.prototxt"
ageModel = "./cascade/age_net.caffemodel"
genderProto = "./cascade/gender_deploy.prototxt"
genderModel = "./cascade/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)


#入力された配列から性別を特定
def detected_gender(bboxes):
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        return gender
# Open a video file or an image file or a camera stream
cap = cv.VideoCapture(0)
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue
    
    #ここでリア充かどうかを判別させる
    if len(bboxes)>1 and len(bboxes)<3:
        first_x1,first_x2,first_y1,first_y2 = bboxes[0]
        second_x1,second_x2,second_y1,second_y2 = bboxes[1]
        first_gender = detected_gender(bboxes[0])
        second_gender = detected_gender(bboxes[1])
        if (abs(first_x1-second_x2) <= 10) and ((first_gender=="Famale" and second_gender=="Male")or(first_gender=="Famale" and second_gender=="Male")):
            print('こいつらはリア充です')
        else:
            print('リア充ではありません。')

    for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        print("Gender:{}".format(gender))
        label = "{}".format(gender)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)