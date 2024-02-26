import cv2
import os

def video_to_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the video frame by frame
    frame_count = 0
    while True:
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Save the frame as an image
        frame_filename = f"{output_folder}/frame_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    fileName = "GrapesOfWrathTrailerClip.mp4"
    video_path = "./videos/" +fileName
    output_folder = "./frames/" +fileName[:-4]+ "/output"
    # print(video_path)
    # print(output_folder)

    video_to_frames(video_path, output_folder)
