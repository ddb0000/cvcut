import os
import sys
import time

# Check if input file is provided
if len(sys.argv) < 2:
    print("Usage: python remove_blue_frames.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = os.path.splitext(input_file)[0] + "_no_blue" + os.path.splitext(input_file)[1]

print(f"Processing file: {input_file}")
start_time = time.time()

# FFmpeg command to remove blue frames
ffmpeg_command = (
    f"ffmpeg -i {input_file} -vf "
    f"\"select='not(gte(b,0.9))',setpts=N/FRAME_RATE/TB\" "
    f"-af \"asetpts=N/SR/TB\" {output_file}"
)

# Execute the command
os.system(ffmpeg_command)

print(f"Done! Output saved as: {output_file}")
print(f"Processing time: {time.time() - start_time:.2f} seconds")
