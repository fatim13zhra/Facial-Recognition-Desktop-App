import cv2
import os
import numpy as np
from PIL import Image
import pickle
def face_reco() :
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #curent directory
    image_dir = os.path.join(BASE_DIR, "trainning")

    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]
                # opens the image file using the PIL (Python Imaging Library) module,
                # converts it to grayscale using the "L" mode.
                pil_images = Image.open(path)
                # unsigned integer 8-bit and is a data type commonly used to represent pixel values in grayscale images.
                # The uint8 data type is suitable for grayscale images because it efficiently stores the intensity information in a single channel,
                # using 8 bits (1 byte) per pixel.
                image_array = np.array(pil_images, "uint8")
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=1)

                for (x, y, w, h) in faces:
                    face = image_array[y:y + h, x:x + w]
                    x_train.append(face)
                    y_labels.append(id_)

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")
