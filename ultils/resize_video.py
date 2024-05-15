from moviepy.editor import VideoFileClip


def resize_video(input_file, output_file, new_dimensions=(640, 640)):
    clip = VideoFileClip(input_file)
    resized_clip = clip.resize(new_dimensions)
    resized_clip.write_videofile(output_file, codec='libx264')
