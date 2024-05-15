import cv2

from ultils import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters
import constants
import numpy as np


class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 600
        self.buffer = 50
        self.padding_court = 20
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_mini_court_drawing_key_points()

    def set_mini_court_drawing_key_points(self):
        drawing_key_points = [0] * 24

        # Draw point 8
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # Draw point 6
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # Draw point 7
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = int(self.court_start_y) + convert_meters_to_pixel_distance(constants.LINE_HEIGHT,
                                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                                           self.drawing_rectangle_width)
        # Draw point 9
        drawing_key_points[6] = drawing_key_points[4] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        # Draw point 5
        drawing_key_points[8] = int((drawing_key_points[0] + drawing_key_points[2]) / 2)
        drawing_key_points[9] = drawing_key_points[1]
        # Draw point 4
        drawing_key_points[10] = int((drawing_key_points[4] + drawing_key_points[6]) / 2)
        drawing_key_points[11] = drawing_key_points[5]
        # Draw point 1
        drawing_key_points[12] = drawing_key_points[0] + convert_meters_to_pixel_distance(
            constants.DOUBLE_ALLY_DIFFERENECE, constants.DOUBLE_LINE_WIDTH, self.drawing_rectangle_width)
        drawing_key_points[13] = drawing_key_points[1]
        # Draw point 3
        drawing_key_points[14] = drawing_key_points[2] - convert_meters_to_pixel_distance(
            constants.DOUBLE_ALLY_DIFFERENECE, constants.DOUBLE_LINE_WIDTH, self.drawing_rectangle_width)
        drawing_key_points[15] = drawing_key_points[1]
        # Draw point 2
        drawing_key_points[16] = drawing_key_points[4] + convert_meters_to_pixel_distance(
            constants.DOUBLE_ALLY_DIFFERENECE, constants.DOUBLE_LINE_WIDTH, self.drawing_rectangle_width)
        drawing_key_points[17] = drawing_key_points[5]
        # Draw point 0
        drawing_key_points[18] = drawing_key_points[6] - convert_meters_to_pixel_distance(
            constants.DOUBLE_ALLY_DIFFERENECE, constants.DOUBLE_LINE_WIDTH, self.drawing_rectangle_width)
        drawing_key_points[19] = drawing_key_points[5]

        # Half line points
        drawing_key_points[20] = drawing_key_points[0]
        drawing_key_points[21] = int((drawing_key_points[1] + drawing_key_points[5])/2)

        drawing_key_points[22] = drawing_key_points[2]
        drawing_key_points[23] = int((drawing_key_points[3] + drawing_key_points[7])/2)

        self.drawing_key_points = drawing_key_points

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

    def draw_badminton_points(self, frame):
        for i, j in zip(range(0, len(self.drawing_key_points), 2), range(int(len(self.drawing_key_points) / 2))):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(j), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0, (0, 0, 255), 0, cv2.LINE_AA)

        return frame

    def draw_court_line(self, frame):
        # Line 0 -> 1
        start_point = (int(self.drawing_key_points[0]), int(self.drawing_key_points[1]))
        end_point = (int(self.drawing_key_points[2]), int(self.drawing_key_points[3]))
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
        # Line 1 -> 3
        start_point = (int(self.drawing_key_points[2]), int(self.drawing_key_points[3]))
        end_point = (int(self.drawing_key_points[6]), int(self.drawing_key_points[7]))
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
        # Line 3 -> 2
        start_point = (int(self.drawing_key_points[6]), int(self.drawing_key_points[7]))
        end_point = (int(self.drawing_key_points[4]), int(self.drawing_key_points[5]))
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
        # Line 2 -> 0
        start_point = (int(self.drawing_key_points[4]), int(self.drawing_key_points[5]))
        end_point = (int(self.drawing_key_points[0]), int(self.drawing_key_points[1]))
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        start_point = (int(self.drawing_key_points[20]), int(self.drawing_key_points[21]))
        end_point = (int(self.drawing_key_points[22]), int(self.drawing_key_points[23]))
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), 2, cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self, frames):
        output = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_badminton_points(frame)
            frame = self.draw_court_line(frame)
            output.append(frame)

        return output
