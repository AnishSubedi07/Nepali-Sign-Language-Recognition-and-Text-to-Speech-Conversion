import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from utils import CvFpsCalc
from model import KeyPointClassifier


class HandGestureRecognition:
    def __init__(self):
        # Argument parsing
        args = self.get_args()

        self.cap_device = args.device
        self.cap_width = args.width
        self.cap_height = args.height

        self.use_static_image_mode = args.use_static_image_mode
        self.min_detection_confidence = args.min_detection_confidence
        self.min_tracking_confidence = args.min_tracking_confidence

        self.use_brect = True

        # Camera preparation
        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        # Model load
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()

        # Read labels
        with open(
            "model/keypoint_classifier/keypoint_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0].strip() for row in self.keypoint_classifier_labels
            ]

        # FPS Measurement
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)

        # Finger gesture history
        self.finger_gesture_history = deque(maxlen=self.history_length)

        self.mode = 0

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help="cap width", type=int, default=960)
        parser.add_argument("--height", help="cap height", type=int, default=540)

        parser.add_argument("--use_static_image_mode", action="store_true")
        parser.add_argument(
            "--min_detection_confidence",
            help="min_detection_confidence",
            type=float,
            default=0.7,
        )
        parser.add_argument(
            "--min_tracking_confidence",
            help="min_tracking_confidence",
            type=int,
            default=0.5,
        )

        args = parser.parse_args()

        return args

    def process_frame(self):
        # Comment out FPS calculation
        # fps = self.cvFpsCalc.get()

        # Camera capture
        ret, image = self.cap.read()
        if not ret:
            return None, None, None
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        landmark_list = []
        detected_char = ""
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                # Write to the dataset file
                self.logging_csv(self.mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                # Drawing part in the live video feed

                # Drawing the rectangle to bound the hand
                debug_image = self.draw_bounding_rect(
                    self.use_brect, debug_image, brect
                )

                # To draw handlandmarks in the live feed
                # debug_image = self.draw_landmarks(debug_image, landmark_list)

                # Drawing the text info above the rectangle
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                )

                detected_char = self.keypoint_classifier_labels[hand_sign_id]
        else:
            self.point_history.append([0, 0])

        debug_image = self.draw_point_history(debug_image, self.point_history)
        # Comment out FPS and mode display
        # debug_image = self.draw_info(debug_image, fps, self.mode)

        # Create hand points image
        hand_points_image = self.draw_landmarks(
            np.zeros_like(debug_image), landmark_list
        )

        return debug_image, hand_points_image, detected_char

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def logging_csv(self, mode, landmark_list):
        if mode == 0:
            pass
        if mode == 1:
            csv_path = "model/keypoint_classifier/keypoint.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([mode, *landmark_list])
        return

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(
                image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[3]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[3]),
                tuple(landmark_point[4]),
                (255, 255, 255),
                2,
            )

            # Index finger
            cv.line(
                image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[6]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[6]),
                tuple(landmark_point[7]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[7]),
                tuple(landmark_point[8]),
                (255, 255, 255),
                2,
            )

            # Middle finger
            cv.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[10]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (255, 255, 255),
                2,
            )

            # Ring finger
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[14]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[14]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[14]),
                tuple(landmark_point[15]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[14]),
                tuple(landmark_point[15]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[15]),
                tuple(landmark_point[16]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[15]),
                tuple(landmark_point[16]),
                (255, 255, 255),
                2,
            )

            # Little finger
            cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[18]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[18]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[18]),
                tuple(landmark_point[19]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[18]),
                tuple(landmark_point[19]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[19]),
                tuple(landmark_point[20]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[19]),
                tuple(landmark_point[20]),
                (255, 255, 255),
                2,
            )

            # Palm
            cv.line(
                image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[0]),
                tuple(landmark_point[1]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[1]),
                tuple(landmark_point[2]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[5]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[9]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[13]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[17]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[17]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[0]),
                (255, 255, 255),
                2,
            )

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 1:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 2:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 3:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 4:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

            if index == 5:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 6:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 7:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 8:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

            if index == 9:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 10:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 11:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 12:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

            if index == 13:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 14:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 15:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 16:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

            if index == 17:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 18:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 19:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

            if index == 20:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(
                image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1
            )

        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text):
        cv.rectangle(
            image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1
        )

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ":" + hand_sign_text  # text for gesture recognition

        # Use PIL to render the text within the black rectangle
        image = self.draw_text_with_pil(
            image,
            info_text,
            (brect[0] + 5, brect[1] - 20),
            "C:/My Files/My Project/Hand Sign Recognition/Noto_Sans_Devanagari/NotoSansDevanagari-VariableFont_wdth,wght.ttf",
            24,
            (255, 255, 255),
        )

        return image

    def draw_text_with_pil(self, image, text, position, font_path, font_size, color):
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        return np.array(img_pil)

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(
                    image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2
                )

        return image

    def draw_info(self, image, fps, mode):
        # Comment out FPS display
        # cv.putText(
        #     image,
        #     "FPS:" + str(fps),
        #     (10, 30),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 0, 0),
        #     4,
        #     cv.LINE_AA,
        # )
        # cv.putText(
        #     image,
        #     "FPS:" + str(fps),
        #     (10, 30),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (255, 255, 255),
        #     2,
        #     cv.LINE_AA,
        # )

        mode_string = ["Logging Key Point", "Logging Point History"]
        if 1 <= mode <= 2:
            cv.putText(
                image,
                "MODE:" + mode_string[mode - 1],
                (10, 90),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
        return image
