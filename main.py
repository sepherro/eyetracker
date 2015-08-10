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

    def _normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v/norm

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
            x_derivative = cv2.filter2D(roi_gray_blurred, cv2.CV_32F, dx)
            y_derivative = cv2.filter2D(roi_gray_blurred, cv2.CV_32F, dy)
            # magic starts here
            for (ex, ey, ew, eh) in self.eyes:
                for outer_cols in xrange(ex, ex+ew):
                    for outer_rows in xrange(ey, ey+eh):
                        response_matrix = np.zeros((ew, eh))
                        for inner_cols in xrange(ex, ex+ew):
                            for inner_rows in xrange(ey, ey+eh):
                                center_vector = [outer_cols - inner_cols, outer_rows - inner_rows]
                                gradient_vector = [x_derivative[inner_cols, inner_rows], y_derivative[inner_cols, inner_rows]]
                                center_vector_norm = self._normalize_vector(center_vector)
                                gradient_vector_norm = self._normalize_vector(gradient_vector)
                                response_raw = np.dot(center_vector_norm, gradient_vector_norm)
                                response_normalized = (float(255 - roi_gray_blurred[inner_cols, inner_rows])/255) * response_raw
                                response_matrix[inner_cols-ex, inner_rows-ey] = response_normalized

                    response_matrix_disp = (response_matrix/np.max(response_matrix))
                    cv2.imshow("pupil", response_matrix_disp)
                    cv2.waitKey(1)



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
