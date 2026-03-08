# Screen Contour Detection Application

import cv2
import numpy as np

class ScreenContourDetector:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, frame, contours):
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            contours = self.process_frame(frame)
            self.draw_contours(frame, contours)
            cv2.imshow('Screen Contour Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = ScreenContourDetector()
    detector.run()