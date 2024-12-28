import cv2
import pickle
import os
import time
from face_reco import face_reco

def face_detection():
    face_reco()
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainner.yml')

    def labels_fun(folder_name):
        path = folder_name
        labels = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        labels = {element: index for index, element in enumerate(labels)}
        return labels

    labels = labels_fun('trainning')
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in labels.items()}

    cap = cv2.VideoCapture(0)

    while (True):
        # capturing frames
        # measure processing time for each frame
        start_time = time.time()
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            face_color = frame[y:y + h, x:x + w]

            id_, conf = recognizer.predict(face_gray)
            accuracy = conf
            if conf >= 50:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y + h), font, 1, color, stroke, cv2.LINE_AA)
                cv2.putText(frame, str("{:.2f}".format(round(accuracy, 2))) + '%', (x, y), font, 1, (0, 255, 0), stroke,
                            cv2.LINE_AA)
                end_time = time.time()
                processing_time = round(end_time - start_time, 2)
                print('Detecting {} took : '.format(name), processing_time, "seconds")
            elif conf < 50:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = "Unrecognized"
                color = (0, 0, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "face.png"
            cv2.imwrite(img_item, face_color)
            # drawing a rectangle
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # displaying frames
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
