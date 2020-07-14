import glob
import os
import cv2

# path of our input videos
videos_path = "D:\\CECILIA\\Desktop\\Vision and Cognitive Systems\\progetto-vision\\Videos"
# path of our output folder where we save the extracted frames
output_path = "D:\\CECILIA\\Desktop\\Vision and Cognitive Systems\\progetto-vision\\images"
os.makedirs(output_path)
# list of all the videos taken from videos_path
videos_list = glob.glob(f'{videos_path}/**/*.*', recursive=False)

for video in videos_list:
    cap = cv2.VideoCapture(video)

    file_path, file_extension = os.path.splitext(video)
    file_name = os.path.basename(file_path)
    frame_num = 0

    # get total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # get video's fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get total seconds of video
    total_seconds = int(total_frames / fps)
    # divide total frames by total seconds to get one frame per second
    # my_frame = int(total_frames / total_seconds)

    # or set the jump of frames = fps, so that I get one frame every fps frames (one frame per second)
    skip_frame = int(fps)

    # for every second of the video, I take one frame every skip_frame
    for i in range(0, total_seconds):
        my_frame = i * skip_frame
        # check for valid frame number
        if my_frame >= 0 & my_frame <= total_frames:
            # set frame position, frame to save
            cap.set(cv2.CAP_PROP_POS_FRAMES, my_frame)
        else:
            break

        ret, frame = cap.read()
        # if MOV files, then rotate 90°
        if file_extension.upper() == ".MOV":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # save the frame in the output path created at the beginning. the image will be saved as VIRB0398-frame-0.png
        cv2.imwrite(
            f'{output_path}/{file_name}-frame-{my_frame}.png', frame)
        print(f"Wrote {file_name}-frame-{my_frame}.png")

    # cap.release()
    cv2.destroyAllWindows()