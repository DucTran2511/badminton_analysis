import os
import pickle
from ultralytics import YOLO
import cv2

class CourtDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def crop_image(self, image):
        results = self.model.predict(image, save_crop=True)
        save_dir = results[0].save_dir
        save_dir += '/crops/court/'
        files = os.listdir(save_dir)
        file_paths = [os.path.join(save_dir, file) for file in files]
        return file_paths[0]

    def resize_image(self, image, size=(640, 640)):

        input_image_path = self.crop_image(image)
        original_image = cv2.imread(input_image_path)
        resized_image = cv2.resize(original_image, size)
        return resized_image

    def detect_frames(self, frames, read_from_stub, stub_path):
        court_list = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                court_list = pickle.load(f)
            return court_list
        for frame in frames:
            court_list.append(self.resize_image(frame))
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(court_list, f)
        return court_list
