import cv2
import os

def frames_to_video(input_folder, output_video_path, fps=30, codec='mp4v'):
    frames = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    frames.sort()

    if not frames:
        print("No frames found in the input folder.")
        return

    frame_path = os.path.join(input_folder, frames[0])
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Failed to read the first frame: {frame_path}")
        return

    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_filename in frames:
        frame_path = os.path.join(input_folder, frame_filename)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Failed to read frame: {frame_path}")
            continue

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to: {output_video_path}")


if __name__ == "__main__":
    folderName = "GrapesOfWrathTrailerClip"
    input_folder = "./frames/"+folderName+"/output"
    output_video_path = "./videos_out/" + folderName+"Color.mp4"

    # print(input_folder)
    # print(output_video_path)

    frames_to_video(input_folder, output_video_path)
