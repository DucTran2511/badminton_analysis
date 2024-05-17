import cv2
import numpy as np
from PIL import Image


class CornerDetector:
    def __init__(self, img_croped):
        self.input = cv2.imread(img_croped)

    def binary_img(self):
        raw_img = cv2.resize(self.input, (640, 640))
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)
        return gray_img, binary_img

    def detect_edges(self):
        # Canny Edge Detection
        gray_img, binary_img = self.binary_img()
        v = np.median(gray_img)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(binary_img, lower, upper, apertureSize=7)
        return edges

    def segment_lines(self, deltaX=280, deltaY=0.5):
        # Hough Transform
        edges = self.detect_edges()
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 90, 90, None, 10, 250)
        h_lines = []
        v_lines = []
        for line in linesP:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) < deltaY:
                    h_lines.append(line)
                elif abs(x2 - x1) < deltaX:
                    v_lines.append(line)

        return h_lines, v_lines

    @staticmethod
    def filterHorizontalLines(h_lines, epsilon=20):
        global upper_line, lower_line
        h_results = []
        min_y = float('inf')
        max_y = float('-inf')
        for segment in h_lines:
            for x1, y1, x2, y2 in segment:
                if y1 < min_y:
                    min_y = y1
                    upper_line = segment
                if y1 > max_y:
                    max_y = y1
                    lower_line = segment
        upper_line[0][1] += epsilon
        upper_line[0][3] += epsilon
        h_results.append(upper_line)
        h_results.append(lower_line)
        return h_results

    @staticmethod
    def find_intersection(line1, line2):
        # extract points
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        # compute determinant
        Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        return Px, Py

    @staticmethod
    def cluster_points(points, nclusters=10):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        return centers

    def find_intersection_points(self):
        Px = []
        Py = []
        h_lines, v_lines = self.segment_lines()
        filtered_h_lines = self.filterHorizontalLines(h_lines)
        for h_line in filtered_h_lines:
            for v_line in v_lines:
                px, py = self.find_intersection(h_line, v_line)
                Px.append(px)
                Py.append(py)
        return Px, Py

    def find_key_point_on_court(self):
        Px, Py = self.find_intersection_points()
        P = np.float32(np.column_stack((Px, Py)))
        nclusters = 10
        centers = self.cluster_points(P, nclusters)
        return centers

    def convert_coordiante_size(self, bbox):
        w, h = self.input.shape[0], self.input.shape[1]
        centers = self.find_key_point_on_court()
        for center in centers:
            center[0] = (center[0] * (h / 640))
            center[1] = (center[1] * (w / 640))
        centers = centers + np.array([bbox[0][0],  bbox[0][1]], dtype=np.float32)
        return centers

    def draw_points(self, centers, frame):
        for i in range(len(centers) - 1):
            x, y = centers[i][0], centers[i][1]
            cv2.putText(frame, str(i), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        return frame

    def draw_key_points_on_videos(self, video_frames, bbox):
        output_video_frames = []
        centers = self.convert_coordiante_size(bbox)
        for frame in video_frames:
            frame = self.draw_points(centers, frame)
            output_video_frames.append(frame)
        return output_video_frames
