from ultralytics import YOLO
import cv2


class CornerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def xy_tracking(self, input_path):
        results = self.model.predict(input_path)[0]
        xy_corner_list = []
        for key_point in results.keypoints:
            result = key_point.xy.tolist()[0]
            xy_corner_list.append(result)
        return xy_corner_list

    def draw_key_points(self, image, xy_corner):
        for i, corner in zip(range(0, 4), xy_corner[0]):
            x, y = xy_corner[0][i][0], xy_corner[0][i][1]
            cv2.putText(image, str(i), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        return image

    def draw_key_points_on_videos(self, video_frames, key_points):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_key_points(frame, key_points)
            output_video_frames.append(frame)
        return output_video_frames
