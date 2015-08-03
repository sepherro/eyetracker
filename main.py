import numpy as np
import cv2


class EyeTracker:
    def __init__(self):
        self.display_results = True
        self.camera_number = 0
        self.video_capture = cv2.VideoCapture(self.camera_number)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.face = None
        self.eyes = None
        if self.display_results:
            cv2.namedWindow('results')

    def get_face_pos(self):
        ret, img = self.video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 15)
        for (x, y, w, h) in faces:
            if self.display_results:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.2, 10)
            if self.display_results:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        if self.display_results:
            cv2.imshow('results', img)

    def cleanup(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

    def get_pupil_pos(self):
        pass

if __name__ == "__main__":
    tracker = EyeTracker()
    while True:
        tracker.get_face_pos()
        # break if ESC pressed
        if int(cv2.waitKey(5)) == 27:
            print "ESC pressed, closing application..."
            tracker.cleanup()
            break