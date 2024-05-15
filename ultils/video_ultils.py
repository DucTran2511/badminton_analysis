import cv2
import os


def read_video(video_path):
    if not os.path.isfile(video_path):
        print(f"Error: The file '{video_path}' does not exist")
        exit()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        print("No frames to save. Aborting.")
        return
    # Get the shape of the frames
    height, width, _ = output_video_frames[0].shape
    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    # We specify output file name (output_video_path), codec 'mp4v', frames per second as 30.0, and frame size as (width, height)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    try:
        # Write each frame to the output file
        for frame in output_video_frames:
            out.write(frame)

        print(f'Video is saved to {output_video_path}')
    finally:
        # Release the VideoWriter
        out.release()

