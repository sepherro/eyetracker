import numpy as np
import cv2


class EyeTracker:
    def __init__(self):
        self.camera_number = 0
        self.video_capture = cv2.VideoCapture(self.camera_number)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.faces = None
        self.eyes = None
        self.image = None
        self.gray = None

    def grab_convert_frame(self):
        ret, img = self.video_capture.read()
        self.image = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_face_pos(self):
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 15)
        print self.faces
        # faces are detected as a list of vectors [x_ulc, y_ulc, f_width, f_height] - ulc -upper left corner

    def get_eye_pos(self):
        for (x, y, w, h) in self.faces:
            roi_gray = self.gray[y:y + h, x:x + w]
            self.eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.2, 10)

    def show_results(self):
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = self.image[y:y + h, x:x + w]
            for (ex, ey, ew, eh) in self.eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.imshow('results', self.image)

    def get_pupil_pos(self):
        # Prewitt filter masks
        dx = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0], ])
        dy = np.transpose(dx)
        # filter with Gaussian
        for (x, y, w, h) in self.faces:
            roi_gray = self.gray[y:y + h, x:x + w]
            roi_gray_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0.05*h, 0.05*w)
            cv2.imshow('roi', roi_gray_blurred)
        pass

    def cleanup(self):
        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = EyeTracker()
    while True:
        tracker.grab_convert_frame()
        tracker.get_face_pos()
        tracker.get_eye_pos()
        tracker.show_results()
        tracker.get_pupil_pos()
        # break if ESC pressed
        if int(cv2.waitKey(5)) == 27:
            print "ESC pressed, closing application..."
            tracker.cleanup()
            break
