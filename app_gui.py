import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
from app import HandGestureRecognition
from collections import deque
import time
import requests
import os
from playsound import playsound


class HandGestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nepali Hand Sign Recognition")

        # Frame for top part (webcam capture and hand signs image) with black border
        self.top_frame = Frame(root, bg="black")
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Inner frame for top part content
        self.inner_top_frame = Frame(self.top_frame, bg="white")
        self.inner_top_frame.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # Webcam Capture Label
        self.webcam_label = Label(self.inner_top_frame)
        self.webcam_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Detected Hand Points Label
        self.hand_points_label = Label(self.inner_top_frame)
        self.hand_points_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Frame for bottom part (text box and buttons) with black border
        self.bottom_frame = Frame(root, bg="black")
        self.bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Inner frame for bottom part content
        self.inner_bottom_frame = Frame(self.bottom_frame, bg="white")
        self.inner_bottom_frame.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # Frame for Detected Characters Label and Text Widget
        self.detected_chars_frame = Frame(self.inner_bottom_frame, bg="white")
        self.detected_chars_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Detected Characters Label
        self.detected_chars_label = Label(
            self.detected_chars_frame,
            text="Detected Characters: ",
            font=("Arial", 14),
            bg="white",
        )
        self.detected_chars_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Detected Characters Text Widget
        self.detected_chars_text = tk.Text(
            self.detected_chars_frame,
            font=("Arial", 14),
            wrap=tk.WORD,
            height=2,
            width=100,
            state=tk.DISABLED,
            relief="solid",
            borderwidth=2,
        )
        self.detected_chars_text.pack(side=tk.LEFT, padx=10, pady=10)
        self.detected_chars_text.insert(tk.END, "")

        # StringVar to manage the text content
        self.detected_chars_var = tk.StringVar()
        self.detected_chars_var.set("")

        # Frame for Clear, Backspace, and Speak Buttons
        self.button_frame = Frame(self.inner_bottom_frame, bg="white")
        self.button_frame.pack(side=tk.TOP, padx=10, pady=0)

        # Clear Button
        self.clear_button = Button(
            self.button_frame, text="Clear", command=self.clear_detected_chars
        )
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Backspace Button
        self.backspace_button = Button(
            self.button_frame, text="Backspace", command=self.backspace_detected_chars
        )
        self.backspace_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Speak Button
        self.speak_button = Button(
            self.button_frame, text="Speak", command=self.speak_detected_chars
        )
        self.speak_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Initialize Hand Gesture Recognition
        self.hand_gesture_recognition = HandGestureRecognition()

        # Initialize deque to store timestamps of detected gestures
        self.gesture_timestamps = deque(maxlen=1)

        # Delay in seconds before registering another gesture
        self.gesture_delay = 3.0  # 3 seconds delay for the gesture to register

        # Load the hand signs image and resize it
        self.hand_signs_image = Image.open("./Pictures/hand_signs.png")
        self.hand_signs_image = self.hand_signs_image.resize(
            (650, 620), Image.Resampling.LANCZOS
        )
        self.hand_signs_image = ImageTk.PhotoImage(self.hand_signs_image)

        # Start the GUI update loop
        self.update_gui()

    def clear_detected_chars(self):
        self.detected_chars_var.set("")
        # Temporarily enable the Text widget to clear its content
        self.detected_chars_text.config(state=tk.NORMAL)
        self.detected_chars_text.delete(1.0, tk.END)
        self.detected_chars_text.config(state=tk.DISABLED)

    def backspace_detected_chars(self):
        current_text = self.detected_chars_var.get()
        if current_text:
            new_text = current_text[:-1]  # Remove the last character
            self.detected_chars_var.set(new_text)
            # Temporarily enable the Text widget to update its content
            self.detected_chars_text.config(state=tk.NORMAL)
            self.detected_chars_text.delete(1.0, tk.END)
            self.detected_chars_text.insert(tk.END, new_text)
            self.detected_chars_text.config(state=tk.DISABLED)

    # Method for audio
    def speak_detected_chars(self):
        current_text = self.detected_chars_var.get().strip()
        print(f"Speaking: {current_text}")
        self.speak_nepali_text(current_text)

    # Method to get google text to speech API
    def speak_nepali_text(self, text):
        url = "https://translate.google.com/translate_tts"
        params = {
            "ie": "UTF-8",
            "q": text,
            "tl": "ne",
            "client": "tw-ob",
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            with open("output.mp3", "wb") as file:
                file.write(response.content)
            playsound("output.mp3")
            os.remove("output.mp3")
        else:
            print("Failed to fetch audio from Google Text-to-Speech API")

    def update_gui(self):
        # Get the latest frame and detected hand points
        frame, hand_points, detected_char = (
            self.hand_gesture_recognition.process_frame()
        )

        if frame is not None:
            # Resize the frame to a smaller size (e.g., 640x480)
            frame = cv.resize(frame, (700, 540))

            # Convert OpenCV image to PIL image
            img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)

        # Display the resized hand signs image
        self.hand_points_label.imgtk = self.hand_signs_image
        self.hand_points_label.configure(image=self.hand_signs_image)

        if detected_char is not None:
            # Check if the delay has passed since the last detected gesture
            current_time = time.time()
            if (
                len(self.gesture_timestamps) == 0
                or (current_time - self.gesture_timestamps[0]) >= self.gesture_delay
            ):
                # Append the new detected character to the existing text
                current_text = self.detected_chars_var.get()

                # Check if the detected character is a space gesture (e.g., gesture ID "space")
                if detected_char == "space":
                    new_text = current_text + " "
                else:
                    new_text = current_text + detected_char

                # Update the detected characters
                self.detected_chars_var.set(new_text)
                # Temporarily enable the Text widget to update its content
                self.detected_chars_text.config(state=tk.NORMAL)
                self.detected_chars_text.delete(1.0, tk.END)
                self.detected_chars_text.insert(tk.END, new_text)
                self.detected_chars_text.config(state=tk.DISABLED)

                # Update the timestamp of the last detected gesture
                self.gesture_timestamps.append(current_time)

        # Schedule the next update
        self.root.after(10, self.update_gui)


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureGUI(root)
    root.mainloop()
