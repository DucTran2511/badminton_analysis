from ultils.video_ultils import read_video, save_video
from trackers import PlayerTracker, ShuttlecockTracker
from corner_badminton_court_detector import CornerDetector, CourtDetector
from mini_court import MiniCourt
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # Read video
    input_video_path = 'D:\Code\Python\Data-Mining/final project\input\input videos/video.mp4'
    video_frames = read_video(input_video_path)

    # Detect player
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    player_detections = player_tracker.detect_frames(frames=video_frames,
                                                     read_from_stub=True,
                                                     stub_path='D:\Code\Python\Data-Mining/final project/tracker_stubs/player_detections.pkl')
    # Detect shuttlecock
    shuttlecock_tracker = ShuttlecockTracker(
        model_path='D:\Code\Python\Data-Mining/final project\model\shuttlecock_detector_model/best.pt')
    shuttlecock_detections = shuttlecock_tracker.detect_frames(frames=video_frames,
                                                               read_from_stub=True,
                                                               stub_path='D:\Code\Python\Data-Mining/final project/tracker_stubs/ball_detections.pkl')
    # Detect court
    court_tracker = CourtDetector('D:\Code\Python\Data-Mining/final project\model\corner_detector_model/best.pt')
    court_detections = court_tracker.crop_image(video_frames[0])
    # Detect corner
    corner_detector = CornerDetector(court_detections)
    # Key points
    key_points = corner_detector.convert_coordiante_size(court_tracker.get_bbox())
    # Track player
    player_detections = player_tracker.player_tracking(key_points, player_detections)
    # Mini Court
    mini_court = MiniCourt(video_frames[0])
    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    # Draw player shuttle boxes
    output_video_frames = shuttlecock_tracker.draw_bboxes(output_video_frames, shuttlecock_detections)
    # Draw corner of badminton court
    output_video_frames = corner_detector.draw_key_points_on_videos(output_video_frames,
                                                                    court_tracker.get_bbox())
    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(video_frames)
    # Save video
    save_video(output_video_frames, 'D:\Code\Python\Data-Mining/final project\output_videos/output_video8.avi')


if __name__ == "__main__":
    main()
