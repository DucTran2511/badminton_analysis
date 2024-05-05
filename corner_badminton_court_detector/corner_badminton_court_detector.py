from ultralytics import YOLO


class CornerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
