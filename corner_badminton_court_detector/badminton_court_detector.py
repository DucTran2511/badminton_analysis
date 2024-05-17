import os
import pickle
from ultralytics import YOLO
import cv2

class CourtDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def crop_image(self, image):
        self.results = self.model.predict(image, save_crop=True)
        save_dir = self.results[0].save_dir
        save_dir += '/crops/court/'
        files = os.listdir(save_dir)
        file_paths = [os.path.join(save_dir, file) for file in files]
        return file_paths[0]

    def get_bbox(self):
        return self.results[0].boxes.xyxy.tolist()
