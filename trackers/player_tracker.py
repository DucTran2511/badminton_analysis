from ultralytics import YOLO
import cv2
import pickle
from ultils import center_bbox
import math
import numpy as np

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def player_tracking(self, centers, people_on_frames):
        people = people_on_frames[0]
        players = self.track_players_ID(centers, people)
        player_detections = []

        for player_dict in people_on_frames:
            player = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in players}
            player_detections.append(player)

        print(player_detections)
        print('hehe')
        return player_detections

    # Player tracking
    def track_players_ID(self, court_keypoint, people):
        upper_point = court_keypoint[5]
        lower_point = court_keypoint[4]
        person_centers = []
        players_id = np.zeros(2)
        for id_tracked, person_bbox in people.items():
            person_centers.append([id_tracked, center_bbox(person_bbox)])

        min_upper_distance = float('inf')
        min_lower_distance = float('inf')
        # Get upper player
        for player_id, person_center in person_centers:
            upper_distance = math.dist(person_center, upper_point)
            lower_distance = math.dist(person_center, lower_point)
            if upper_distance < min_upper_distance:
                min_upper_distance = upper_distance
                players_id[0] = player_id
            if lower_distance < min_lower_distance:
                min_lower_distance = lower_distance
                players_id[1] = player_id
        return players_id

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result

        return player_dict

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames

