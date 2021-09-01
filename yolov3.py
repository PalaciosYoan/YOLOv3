"""Author: Yoan Palacios"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320
confidentsThreshold = 0.5
nmsThreshold = 0.3 #lower < boxes

classFile = r'./traningStuff/coco.names'
classNames = []

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


modelConfig = './traningStuff/yolov3-320.cfg'
modelWeights = './traningStuff/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confidentsVal = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidenceVal = scores[classId]
            if confidenceVal > confidentsThreshold:
                w,h = int(detection[2]*wT), int(detection[3]*hT )
                x,y = int(detection[0]*wT - w/2), int(detection[1]*hT - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confidentsVal.append(float(confidenceVal))
    
    indices = cv2.dnn.NMSBoxes(bbox, confidentsVal, confidentsThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h =box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confidentsVal[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
     
    outputs = net.forward(outputNames)
    

    findObjects(outputs, img)
    

    cv2.imshow('Img', img)
    cv2.waitKey(1)

