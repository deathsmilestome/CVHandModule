import cv2
import mediapipe as mp
import time


# 640x480 cam
class HandDet:
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detection_conf,
                                         self.track_conf, )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_dots(self, img, hand_N=0, draw=False):
        lms_list = []
        if self.results.multi_hand_landmarks:
            hand_lms = self.results.multi_hand_landmarks[hand_N]
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lms_list.append([id, cx, cy])  # add "if id ==" construction for your needed dots
                if draw:  # you can add "and id == " to highlight some dots| don't forget to put draw=True in main
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lms_list  # you can return positions for any dot check line 38


def main():
    first_time_stamp = 0
    second_time_stamp = 0
    cap = cv2.VideoCapture(0)
    detector = HandDet()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)  # add draw = False if u don't want to draw dots and connections
        lms_list = detector.find_dots(img)  # add draw = True if u want to highlight dot
        if len(lms_list) != 0:
            print(lms_list[8])  # 8 index_finger_tip, all dots in README

        second_time_stamp = time.time()
        fps = 1 / (second_time_stamp - first_time_stamp)
        first_time_stamp = second_time_stamp

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
