import cv2
import os
import pickle
from tkinter import *
from PIL import Image, ImageTk
import face_recognition
from face_reco import face_reco
import customtkinter
import numpy as np
import subprocess
import datetime

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def haar(self):
        self.model = 'haar'
        return self.model

    def hog(self):
        self.model = 'hog'
        return self.model

    def secured_hog(self):
        self.model = 'secured_hog'
        return self.model

    #datageneration :st
    def generate_dataset(self):
        name=self.entry.get()
        face_classifier = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
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
                file_name_path = new_person + "/" + str(img_id) + ".jpg"  # Fixed the file name path
                cv2.imwrite(file_name_path, face) #saves the grayscale face
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped_Face", face)
                if cv2.waitKey(1) == 13 or int(img_id) == 200:  # Use waitKey() instead of catKey() Enter key (key code 13)
                    new_person_hog = os.path.join("ImagesAttendance", name + ".jpg")
                    cv2.imwrite(new_person_hog, frame)  # Save the original frame instead of the face
                    cv2.imshow("Cropped_Face", face)
                    break

        cap.release()
        cv2.destroyAllWindows()


    def __init__(self):
        super().__init__()
        self.model=None

        self.grid_columnconfigure(1, weight=1) #weight parameter determines how the column should expand or shrink when the window is resized
        self.grid_rowconfigure(0, weight=1)

        # configure window
        self.title("Comparing between face detection models")
        self.geometry(f"{1100}x{580}")
        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(3, weight=0)

        # Create a sidebar frame
        sidebar_frame = Frame(self, width=200, bg="gray")
        sidebar_frame.grid(row=0, column=0, sticky="ns")

        # Add widgets to the sidebar
        sidebar_label = Label(sidebar_frame, text="Sidebar", font=("Arial", 16), bg="gray", fg="white")
        sidebar_label.pack(pady=20)

        # icons
        icon_image = Image.open(os.path.abspath("icons/face-scan-1.png"))
        icon_image = icon_image.resize((35, 35))  # Resize the icon image to desired size
        icon_photo = ImageTk.PhotoImage(icon_image)

        icon_image = Image.open(os.path.abspath("icons/face-id-user.png"))
        icon_image = icon_image.resize((35, 35))
        icon_photo_1 = ImageTk.PhotoImage(icon_image)

        icon_image = Image.open(os.path.abspath("icons/face-id-dark.png"))
        icon_image = icon_image.resize((35, 35))
        icon_photo_2 = ImageTk.PhotoImage(icon_image)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="FACIAL RECOGNITION",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))

        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,command=self.haar, anchor=W, image=icon_photo_2, height=50, width=190, text_color="black", fg_color="#1abc9c", font=('Roboto', 18))
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_1.configure(text="HaarCascade")
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame,command=self.hog, anchor=W, image=icon_photo, height=50, width=190, text_color="black", fg_color="#1abc9c", font=('Roboto', 18))
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_2.configure(text="HOG")
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame,command=self.secured_hog, anchor=W, image=icon_photo_1, height=50, width=190, text_color="black", fg_color="#1abc9c", font=('Roboto', 18))
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_3.configure(text="Secured HOG")
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w", font=('Roboto', 18))
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,values=["Light", "Dark", "System"],command=self.change_appearance_mode_event, height=50, width=190, font=('Roboto', 18), text_color="black")
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w", font=('Roboto', 18))
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],command=self.change_scaling_event, height=50, width=190, font=('Roboto', 18), text_color="black")
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # Create a main frame
        main_frame = Frame(self)
        main_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create a label and display it on app
        label_widget = Label(main_frame)
        label_widget.grid(row=0, column=0, sticky="nsew")

        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
        self.bind('<Escape>', lambda e: self.quit())

        def project(app, vid, label_widget):

            total_frames = 0
            correct_recognitions = 0

            path = "ImagesAttendance"
            images = []
            classNames = []
            myList = os.listdir(path)
            print(myList)
            for cl in myList:
                curImg = cv2.imread(f'{path}/{cl}')
                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])
            print(classNames)

            def findEncodings(imgList):
                encodeList = []
                for img in imgList:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encode = face_recognition.face_encodings(img)[0]
                    encodeList.append(encode)
                return encodeList

            def markAttendance(name, accuracy):
                nonlocal correct_recognitions

                with open('Attendance.csv', 'r+') as f:
                    myDataList = f.readlines()
                    nameList = []
                    print(myDataList)
                    for line in myDataList:
                        entry = line.split(',')
                        nameList.append(entry[0])
                    if name not in nameList:
                        now = datetime.datetime.now()
                        dtString = now.strftime("%H:%M:%S")
                        f.writelines(f'\n{name},{dtString}')
                        correct_recognitions += 1
                return accuracy

            print("Encoding started ...")
            encodeListKnown = findEncodings(images)
            print("Encoding Complete.")
            print("Total of encoded images : ", len(encodeListKnown))

            vid = cv2.VideoCapture(0)

            def show_frame():
                nonlocal vid
                nonlocal label_widget
                nonlocal total_frames
                nonlocal correct_recognitions

                total_frames += 1

                success, img = vid.read()
                """Reduce the size of the image, to speed up the process"""
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                facesCurFrame = face_recognition.face_locations(imgS)

                encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
                    now = datetime.datetime.now()
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        accuracy = 100 - faceDis[matchIndex] * 100  # Calculate accuracy
                        accuracy = round(accuracy, 2)  # Round accuracy to 2 decimal places
                        markAttendance(name, accuracy)  # Call markAttendance() with name and accuracy
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, f'{name} {accuracy}%', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        end_time = datetime.datetime.now()
                        processing_time = (end_time - now).total_seconds()
                        processing_time = round(processing_time, 2)
                        cv2.putText(img, 'Detection time : ' + str(processing_time) + 's', (x1- 6, y2-y1+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)
                label_widget.config(image=img)
                label_widget.image = img

                accuracy = (correct_recognitions / total_frames) * 100
                print("Accuracy:", accuracy)

                app.after(1, show_frame)

            show_frame()
            accuracy = (correct_recognitions / total_frames) * 100


        def face_detection(vid, label_widget):
            face_reco()
            face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainner.yml')

            def labels(folder_name):
                path = folder_name  # replace with the path to your folder
                labels = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                labels = {element: index for index, element in enumerate(labels)}
                return labels

            labels = labels('trainning')
            with open("labels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                labels = {v: k for k, v in labels.items()}

            def detect_faces():
                # capturing frames
                # measure processing time for each frame
                start_time = datetime.datetime.now()
                ret, frame = vid.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                for (x, y, w, h) in faces:
                    face_gray = gray[y:y + h, x:x + w]
                    face_color = frame[y:y + h, x:x + w]

                    id_, conf = recognizer.predict(face_gray)
                    accuracy = conf
                    if conf >= 50:
                        cv2.putText(frame, labels[id_], (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                        cv2.putText(frame, str("{:.2f}".format(round(accuracy, 2))) + '%', (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        end_time = datetime.datetime.now()
                        processing_time = (end_time - start_time).total_seconds()
                        processing_time = round(processing_time, 2)
                        print('Detecting {} took: '.format(labels[id_]), processing_time, "seconds")
                        cv2.putText(frame, 'Detection time : '+str(processing_time) + 's', (100, 700),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    elif conf < 50:
                        cv2.putText(frame, "Unrecognized", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)

                    img_item = "face.png"
                    cv2.imwrite(img_item, face_color)
                    # drawing a rectangle
                    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

                # Convert image from one color space to another
                opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

                # Capture the latest frame and transform it into an image
                captured_image = Image.fromarray(opencv_image)

                # Convert the captured image to a PhotoImage
                photo_image = ImageTk.PhotoImage(image=captured_image)

                # Display the PhotoImage in the label
                label_widget.photo_image = photo_image

                # Configure the image in the label
                label_widget.configure(image=photo_image)

                # Repeat the same process after every 10 milliseconds
                label_widget.after(10, detect_faces)

            detect_faces()

        def open_camera():

            if self.model=="haar":
                face_detection(vid, label_widget)
            elif self.model=="hog":
                project(app, vid, label_widget)
                #print("i'm hog")
                #subprocess.Popen(["python", os.path.join(os.getcwd(), "test2.py")])
            elif self.model=="secured_hog":
                subprocess.Popen(["python", os.path.join(os.getcwd(), "main.py")])
                print("i'm secured hog")
            else :
                # Capture the video frame by frame
                _, frame = vid.read()

                # Convert image from one color space to other
                opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

                # Capture the latest frame and transform to image
                captured_image = Image.fromarray(opencv_image)

                # Convert captured image to photoimage
                photo_image = ImageTk.PhotoImage(image=captured_image)

                # Displaying photoimage in the label
                label_widget.photo_image = photo_image

                # Configure image in the label
                label_widget.configure(image=photo_image)

                # Repeat the same process after every 10 seconds
                label_widget.after(10, open_camera)




        ###This section is related to opening the camera
        button1 = customtkinter.CTkButton(main_frame, text="Click here to open camera", fg_color="transparent", border_width=0, text_color=("black", "black"), command=open_camera, hover_color="#EA2027", font=('Roboto', 18))
        button1.grid(row=1, column=0, pady=(10, 0))


        #This section is related to data generation
        self.entry = customtkinter.CTkEntry(self, placeholder_text="New user ? Enter your name here.")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="#EA2027", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.generate_dataset, font=('Roboto', 18))
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.main_button_1.configure(text="Enter")

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)


        # Create an infinite loop for displaying app on screen
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.mainloop()
