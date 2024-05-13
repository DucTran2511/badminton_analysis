import cv2
import numpy as np
from PIL import Image


class CornerDetector:

    @staticmethod
    def binary_img(image_path):
        raw_img = image_path
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)
        return gray_img, binary_img

    def detect_edges(self, image_path):
        # Canny Edge Detection
        gray_img, binary_img = self.binary_img(image_path)
        v = np.median(gray_img)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(binary_img, lower, upper, apertureSize=7)
        return edges

    def segment_lines(self, image_path, deltaX=280, deltaY=0.5):
        # Hough Transform
        edges = self.detect_edges(image_path)
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
    def filterHorizontalLines(h_lines):
        h_results = []
        for segment in h_lines:
            for x1, y1, x2, y2 in segment:
                if (y1 or y2) > 600:
                    h_results.append(segment)
                if (y1 or y2) < 30:
                    segment[0][1] += 40
                    segment[0][3] += 40
                    h_results.append(segment)
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

    def find_intersection_points(self, img_path):
        Px = []
        Py = []
        h_lines, v_lines = self.segment_lines(img_path)
        filtered_h_lines = self.filterHorizontalLines(h_lines)
        for h_line in filtered_h_lines:
            for v_line in v_lines:
                px, py = self.find_intersection(h_line, v_line)
                Px.append(px)
                Py.append(py)
        return Px, Py

    def find_key_point_on_court(self, img_path):
        Px, Py = self.find_intersection_points(img_path)
        P = np.float32(np.column_stack((Px, Py)))
        nclusters = 10
        centers = self.cluster_points(P, nclusters)
        return centers

    def convert_coordiante_size(self, img, img_path):
        centers = self.find_key_point_on_court(img_path)
        image = Image.fromarray(img)
        weight, height = image.size
        print(weight, height)
        for i in range(len(centers) - 1):
            centers[i][0], centers[i][1] = centers[i][0] * (weight / 640), centers[i][1] * (height / 640)
        return centers

    def draw_points(self, image, img_path):
        centers = self.find_intersection_points(img_path)
        h_lines, v_lines = self.segment_lines(img_path)
        filtered_h_lines = self.filterHorizontalLines(h_lines)
        for i in range(len(centers) - 1):
            x, y = centers[i][0], centers[i][1]
            cv2.putText(image, str(i), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        for i in range(len(v_lines)):
            l = v_lines[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

        # Drawing Horizontal Hough Lines on image
        for i in range(len(filtered_h_lines)):
            l = filtered_h_lines[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
        return image

    def draw_key_points_on_videos(self, video_frames, court_list):
        output_video_frames = []
        for frame, court in zip(video_frames, court_list):
            frame = cv2.resize(frame, (640, 640))
            frame = self.draw_points(frame, court)
            output_video_frames.append(frame)
        return output_video_frames
