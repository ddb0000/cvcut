from moviepy.video.io.VideoFileClip import VideoFileClip

def cut_video(input_file, output_file, start_time=60):
    """
    Removes the first `start_time` seconds from a video using MoviePy.

    Parameters:
        input_file (str): Path to the input video file.
        output_file (str): Path to the output video file.
        start_time (int): Time in seconds to start the output video from.
    """
    try:
        # Load the video
        with VideoFileClip(input_file) as video:
            # Trim the video from the start_time to the end
            trimmed_video = video.subclip(start_time, video.duration)
            # Write the result to the output file
            trimmed_video.write_videofile(output_file, codec="libx264", audio_codec="aac")
            print(f"Video processed successfully. Saved to {output_file}")
    except Exception as e:
        print(f"Error processing video: {e}")

# Example usage
input_file = "Pai 19 anos.mp4"
output_file = "output.mp4"
cut_video(input_file, output_file)
