import cv2
import numpy as np


class YOLO_CV2:
    # PROPERTY
    module_path = "yolo_module/weight/{}"
    confidence = 0
    threshold = 0
    coco_name_path = ""
    net = None
    ln = None

    # METHOD
    def __init__(self, model="yolo3-tiny", confidence=0.2, threshold=0.15):
        self.confidence = confidence
        self.threshold = threshold

        cfg_path = self.module_path.format(model + ".cfg")
        weight_path = self.module_path.format(model + ".weights")
        self.net = cv2.dnn.readNet(cfg_path, weight_path)
        layerNames = self.net.getLayerNames()
        self.ln = [layerNames[i[0] - 1]
                   for i in self.net.getUnconnectedOutLayers()]

        coco_name_path = self.module_path.format("coco.names")
        self.labels = np.array(open(coco_name_path).read().strip().split("\n"))

    def detect(self, frame, pred_classes=["person"]):
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []
        for output in outs:
            for detection in output:
                probs = detection[5:]
                classID = np.argmax(probs)
                confidence = probs[classID]

                if confidence > self.confidence:
                    box = detection[0:4]*np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence, self.threshold)
        if len(idxs) == 0:
            return [], [], []
        
        idxs = idxs.flatten()
        idxs = [idx for idx in idxs if self.labels[classIDs[idx]] in pred_classes]
        boxes = np.array(boxes)[idxs]
        confidences = np.array(confidences)[idxs]
        classIDs = np.array(classIDs)[idxs]
        return boxes, confidences, self.labels[classIDs]
