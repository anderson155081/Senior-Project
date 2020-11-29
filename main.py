import cv2
from blink_detect import f_liveness_detection
import numpy as np
import imutils
import time
import os
import math
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def blink_getimg():
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # inicializar conteo de parpadeos
    COUNTER,TOTAL = 0,0
    input_type = "webcam"
    #----------------------------- Video ------------------------------
    if input_type == "webcam":
        cv2.namedWindow("preview")
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        while True:
            star_time = time.time()
            ret, im = cam.read()
            im = imutils.resize(im, width=720)
            # ingresar flujo de datos
            out = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL)
            TOTAL= out['total_blinks']
            COUNTER= out['count_blinks_consecutives']
            cv2.imshow('preview',im)
            if TOTAL > 3:
                cv2.imwrite('rec.png',im)
                break

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


if __name__ == "__main__":

    blink_getimg()

    predictions = predict("rec.png", model_path="model/trained_knn_model.clf")

    for name, (top, right, bottom, left) in predictions:
        outF = open("predict_result.txt", "w")
        outF.write(name)
    outF.close()

    # Print results on the console
    for name, (top, right, bottom, left) in predictions:
        print("- Found {}".format(name))
