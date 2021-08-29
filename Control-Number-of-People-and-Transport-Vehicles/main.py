import sys
import cv2
import time
import copy
import numpy as np
from module.tools.info_man import InfoManager
from module.tools import generate_detections as gdet
from module.deep_sort.tracker import Tracker
from module.deep_sort.detection import Detection
from module.deep_sort import nn_matching
from module.deep_sort import preprocessing
from yolo_module.yolo_cv2 import YOLO_CV2
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# loading model
MODEL = "yolov3-320"
CONFIDENCE = 0.7  # Threshold confidence probability for detect
THRESHOLD = 0.15  # threshold for NMS
yolo_model = YOLO_CV2(MODEL, CONFIDENCE, THRESHOLD)

# deep_sort
# Definition of the Parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
skip_frame = 5

model_filename = 'models/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
ds_tracker = Tracker(metric)


def get_processed_box(video_link):
    video_capture = cv2.VideoCapture(video_link)
    box = []

    print("Select box for process")
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret == False:
            print("EOF")
        # frame = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
        r = cv2.selectROI("Select BOX", frame)
        cv2.destroyWindow('Select BOX')
        box = [
            (r[0], r[1]),
            (r[0] + r[2], r[1]),
            (r[0] + r[2], r[1] + r[3]),
            (r[0], r[1] + r[3])
        ]
        break
    return box


def allocate_box_follow_class(boxs, classes, predicted_classes, confidences, min_area=400):
    dict_of_boxs = {}
    for index in range(len(boxs)):
        if boxs[index][2] * boxs[index][3] < min_area:
            continue
        if classes[index] not in dict_of_boxs:
            dict_of_boxs[classes[index]] = {
                "bboxes": [],
                "conf": []
            }

        dict_of_boxs[classes[index]]["bboxes"].append(boxs[index])
        dict_of_boxs[classes[index]]["conf"].append(confidences[index])
    return dict_of_boxs


def counting_object(pre_polygon, video_link, pred_classes):
    trackers = {}
    info_mans = {}
    for class_name in predicted_classes:
        # create tracker
        tracker = Tracker(metric)
        trackers[class_name] = copy.deepcopy(tracker)
        del tracker

        # id in-out
        info_man = InfoManager(pre_polygon)
        info_mans[class_name] = info_man
        del info_man

    video_capture = cv2.VideoCapture(video_link)

    while True:
        start = time.time()
        ok, frame = video_capture.read()
        if not ok:
            continue

        # process polygon
        width, height, __ = frame.shape
        for i in range(len(pre_polygon)):
            cv2.line(frame, pre_polygon[i - 1],
                     pre_polygon[i], (0, 0, 255), 2)

        bboxes, confidences, labels = yolo_model.detect(
            frame, pred_classes)
        dict_of_boxes = allocate_box_follow_class(
            bboxes, labels, pred_classes, confidences)

        for class_name in dict_of_boxes:
            tracker = trackers[class_name]
            bboxes = dict_of_boxes[class_name]["bboxes"]
            conf = dict_of_boxes[class_name]["conf"]
            features = encoder(frame, bboxes)

            detections = [Detection(bbox, 1.0, feature)
                          for bbox, feature in zip(bboxes, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            lst_id = []
            for track in tracker.tracks:
                bbox = track.to_tlbr()
                x, y, x2, y2 = map(int, bbox)
                w = x2 - x
                h = y2 - y
                track_id = track.track_id
                lst_id.append(track_id)
                is_in_select_box = info_mans[class_name].update(
                    (int(x+w/2), int(y+h/2)), track_id)

                if is_in_select_box:
                    color = (0, 255, 0)
                    # cv2.putText(
                    #     frame, "{0:.2f} km/h".format(
                    #     info_mans[class_name].info_dict[track_id]["speed"]),
                    #     (x, y), 0, 0.5, (0, 255, 0), 2)
                else:
                    color = (255, 255, 255)

                # cv2.putText(
                #     frame, "{0:.2f}".format(track_id),
                #     (x, y), 0, 0.5, (0, 255, 0), 2)

                cv2.putText(
                    frame, "{}".format(class_name),
                    (x, y), 0, 0.5, (255, 255, 255), 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              color, 2)
                cv2.circle(frame, (int(x+w/2), int(y+h/2)),
                           4, (255, 255, 255), -1)

        class_index = 0
        for class_name in info_mans:
            class_index += 1
            cv2.putText(frame,
                        "{}: {}".format(
                            class_name, info_mans[class_name].count),
                        (100*class_index - 80, 20), 0, 0.5, (0, 255, 0), 2
                        )
        # if len(bboxes) > 0:
        #     for bbox in bboxes:
        #         box_color = (255, 0, 0)
        #         x, y, w, h = bbox
        #         cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        cv2.imshow('Video', frame)
        print("FPS: {0:.2f}".format(1/(time.time()-start)))

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break


if __name__ == "__main__":
    # args
    args = sys.argv
    if len(args) < 3:
        print("python main.py <video_link> <object-classes>")
        print("eg: python main.py video/Road-traffic-video.mp4 person")
        print("eg: python main.py video/Road-traffic-video.mp4 car")
        print("eg: python main.py video/Road-traffic-video.mp4 car,motobike")
        exit()
    video_link = args[1]
    predicted_classes = args[2].split(",")

    pre_polygon = get_processed_box(video_link)

    counting_object(pre_polygon, video_link, predicted_classes)
