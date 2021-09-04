import cv2
import numpy as np

def visualize_detect(bgr_image, bboxes, color = (0, 0, 255), thinkness = 3):
    '''
     Visualize face detection
    '''
    vis_image = bgr_image.copy()
    for box in bboxes:
        pt1 = tuple(map(int, box[0:2]))
        pt2 = tuple(map(int, box[2:4]))
        cv2.rectangle(vis_image, pt1, pt2, color, thinkness)
    return vis_image

def visualize_recognition(bgr_image, text, box, color = (0, 0, 255)):
    '''
     Visualize face recognition
    '''
    pt1 = tuple(map(int, box[0:2]))
    pt2 = tuple(map(int, box[2:4]))
    vis_image = bgr_image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    vis_image = visualize_detect(vis_image, [box], color = (0, 255, 0))
    cv2.putText(vis_image, text, pt1, font, 
                   1, color, 2, cv2.LINE_AA)
    return vis_image