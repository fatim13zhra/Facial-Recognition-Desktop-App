import os.path
import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util
from test import test
import customtkinter as ctk


class App:
    def __init__(self):


        self.main_window = ctk.CTk()
        self.main_window.geometry("950x520+350+100")
        self.main_window.title("Secure face recognition system")

        self.login_button_main_window = util.get_button(self.main_window, 'TEST','transparent', self.login, '#1abc9c', anchor="", image=None)
        self.login_button_main_window.place(x=700, y=100)
        # icons
        icon_image = Image.open(os.path.abspath("icons/house-chimney-1-alternate.png"))
        icon_image = icon_image.resize((35, 35))  # Resize the icon image to desired size
        icon_photo = ImageTk.PhotoImage(icon_image)

        self.logout_button_main_window = util.get_button(self.main_window, '     BACK ', 'transparent',  self.logout, '#EA2027', anchor=tk.W, image=icon_photo, text_color="white")
        self.logout_button_main_window.place(x=700, y=400)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'REGISTER', 'transparent',
                                                                    self.register_new_user, fg='#1289A7', anchor="", image=None)
        self.register_new_user_button_main_window.place(x=700, y=200)

        self.webcam_label = util.get_img_label(self.main_window) #width=700, height=500
        self.webcam_label.place(x=10, y=0)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir='resources/anti_spoof_models',
                device_id=0
                )

        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def logout(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir='/home/phillip/Desktop/todays_tutorial/27_face_recognition_spoofing/code/face-attendance-system/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )

        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Hasta la vista !', 'Goodbye, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')


    def register_new_user(self):
        self.register_new_user_window = ctk.CTkToplevel(self.main_window)
        self.register_new_user_window.geometry("950x520+350+100")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'ACCEPT', "transparent", self.accept_register_new_user, '#1abc9c', anchor="", image=None)
        self.accept_button_register_new_user_window.place(x=700, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'TRY AGAIN', "transparent", self.try_again_register_new_user, fg='#1289A7', anchor="")
        self.try_again_button_register_new_user_window.place(x=700, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        #, width=700, height=500
        self.capture_label.place(x=10, y=0)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=700, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=700, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get()
        #1.0, "end-1c"
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
