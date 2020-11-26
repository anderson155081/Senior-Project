import cv2
import imutils
from blink_detect import f_utils
import dlib
import numpy as np
from blink_detect.blink_detection import f_blink_detection


# instaciar detectores
frontal_face_detector    = dlib.get_frontal_face_detector()
blink_detector           = f_blink_detection.eye_blink_detector() 

eye_landmarks = "blink_detect/blink_detection/model_landmarks/shape_predictor_68_face_landmarks.dat"

def detect_liveness(im,COUNTER=0,TOTAL=0):
    # preprocesar data
    gray = gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # face detection
    rectangles = frontal_face_detector(gray, 0)
    boxes_face = f_utils.convert_rectangles2array(rectangles,im)
    if len(boxes_face)!=0:
        
        # usar solo el rostro con la cara mas grande
        areas = f_utils.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = [list(boxes_face[index])]

        # -------------------------------------- blink_detection ---------------------------------------
        COUNTER,TOTAL = blink_detector.eye_blink(gray,rectangles,COUNTER,TOTAL)
    else:
        TOTAL = 0
        COUNTER = 0
    # -------------------------------------- output ---------------------------------------
    output = {
        'total_blinks': TOTAL,
        'count_blinks_consecutives': COUNTER
    }
    return output

