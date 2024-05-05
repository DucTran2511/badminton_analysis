from ultils.video_ultils import read_video, save_video
from trackers import PlayerTracker, ShuttlecockTracker
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # Read video
    input_picture_path = 'D:\Code\Python\Data-Mining/final project\input\input videos/vi2.mp4'
    video_frames = read_video(input_picture_path)

    # Detect player
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    player_detections = player_tracker.detect_frames(frames=video_frames,
                                                     read_from_stub=True,
                                                     stub_path='D:\Code\Python\Data-Mining/final project/tracker_stubs/player_detections.pkl')
    # Detect shuttlecock
    shuttlecock_tracker = ShuttlecockTracker(model_path='D:\Code\Python\Data-Mining/final project\model\shuttlecock_detector_model/best.pt')
    shuttlecock_detections = shuttlecock_tracker.detect_frames(frames=video_frames,
                                                     read_from_stub=True,
                                                     stub_path='D:\Code\Python\Data-Mining/final project/tracker_stubs/ball_detections.pkl')

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = shuttlecock_tracker.draw_bboxes(output_video_frames, shuttlecock_detections)
    save_video(output_video_frames, 'D:\Code\Python\Data-Mining/final project\output_videos/output_video4.avi')


if __name__ == "__main__":
    main()
