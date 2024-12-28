import cv2
import os

def generate_dataset(name):
    face_classifier = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
        print(faces)
        if len(faces) == 0:  # Check if faces is empty
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)
    img_id = 0

    new_person = os.path.join("trainning", name)

    # Create the new folder
    os.mkdir(new_person)
    while True:
        ret, frame = cap.read()
        cropped = face_cropped(frame)
        if cropped is not None:
            img_id += 1
            face = cv2.resize(cropped, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path =new_person+"/" + str(img_id) + ".jpg"  # Fixed the file name path
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped_Face", face)
            if cv2.waitKey(1) == 13 or int(img_id) == 1000:  # Use waitKey() instead of catKey()
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed")