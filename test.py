#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
from keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/bei1.mp4")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

line = [(0, 600), (1920, 600)]  # 在这里可以修改检测线的两点坐标

# 如果线段AB和CD相交，则返回true
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# 定义序列，存储人的行动轨迹
pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

def main(yolo):
    person_num = 0
    max_cosine_distance = 0.5 #余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3 #非极大抑制的阈值

    counter = []

    #deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc('F','L','V','1')
        out = cv2.VideoWriter('./output/'+args["input"][43:57]+ "_" + args["class"] + '_output.avi', fourcc, 25, (w, h))
        list_file = open('detection.txt', 'w')
        num_file = open('bei1.txt', 'w')
        frame_index = -1

    fps = 0.0

    count_frame = 0
    while True:

        ret, frame = video_capture.read()

        count_frame += 1


        if ret != True:
            break
        t1 = time.time()

        # bgr to rgb
        image = Image.fromarray(frame[...,::-1])

        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # 调用跟踪
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            #cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1

            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))

            pts[track.track_id].append(center)
            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

            # 画路径线
            if pts[track.track_id][len(pts[track.track_id]) - 1] is not None and pts[track.track_id][len(pts[track.track_id]) - 2] is not None:
                cv2.line(frame, (pts[track.track_id][len(pts[track.track_id]) - 1]), (pts[track.track_id][len(pts[track.track_id]) - 2]), (color), thickness)
            # 判断行人两帧之间的连线是否超过画的线，超过进行计数

            if intersect(pts[track.track_id][len(pts[track.track_id]) - 1],
                         pts[track.track_id][len(pts[track.track_id]) - 2],
                         line[0], line[1]) and pts[track.track_id][0][1]<600:
                person_num += 1

            # if intersect(pts[track.track_id][len(pts[track.track_id]) - 1],
            #              pts[track.track_id][len(pts[track.track_id]) - 2],
            #              line[0], line[1]) and pts[track.track_id][len(pts[track.track_id]) - 2][1]>600:
            if intersect(pts[track.track_id][len(pts[track.track_id]) - 1],
                         pts[track.track_id][len(pts[track.track_id]) - 2],
                         line[0], line[1]) and pts[track.track_id][0][1] > 600:
                person_num -= 1
        # draw line
        cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

        count = len(set(counter))
        cv2.putText(frame, "Current Counter: "+str(person_num),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),2)
        #cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        #cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            #save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

            if count_frame == 1500:
                count_frame = 0
                num_file.write(str(person_num)+' ')
                num_file.write('\n')

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
        num_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
