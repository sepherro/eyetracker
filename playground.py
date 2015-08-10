import numpy as np
import cv2


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


img = np.ones((7, 7), dtype=np.uint8)
img = img * 100
img[3, 3] = 50

print img

dx = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], ])
dy = np.transpose(dx)

x_derivative = cv2.filter2D(img, cv2.CV_32F, dx)
y_derivative = cv2.filter2D(img, cv2.CV_32F, dy)

print x_derivative
print y_derivative

print img.shape[0]

for outer_cols in xrange(0, img.shape[1]):
    for outer_rows in xrange(0, img.shape[0]):
        response_matrix = np.zeros(img.shape)
        for inner_cols in xrange(0, img.shape[1]):
            for inner_rows in xrange(0, img.shape[0]):
                center_vector = [outer_cols - inner_cols, outer_rows - inner_rows]
                gradient_vector = [x_derivative[inner_cols, inner_rows], y_derivative[inner_cols, inner_rows]]
                center_vector_norm = normalize_vector(center_vector)
                gradient_vector_norm = normalize_vector(gradient_vector)
                response_raw = np.dot(center_vector_norm, gradient_vector_norm)
                response_normalized = (float(255 - img[inner_cols, inner_rows])/255) * response_raw
                response_matrix[inner_cols, inner_rows] = response_normalized


print response_matrix