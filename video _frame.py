import glob
import os
import re
import shutil

import cv2

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

# path of our input videos. da modificare
videos_path = "D:\\CECILIA\\Desktop\\Vision and Cognitive Systems\\progetto-vision\\Project material\\output"
# path of our output folder where we save the extracted frames. da modificare
output_path = "D:\\CECILIA\\Desktop\\Vision and Cognitive Systems\\progetto-vision\\Project material\\images"
try:
    os.makedirs(output_path)
except:
    print('error')

# list of all the videos taken from videos_path
label_list = []
videos_list = []
for file in os.listdir(videos_path):
    if file.endswith('.txt'):
        label_list.append(file)
    else:
        videos_list.append(file)

label_list.sort(key=natural_keys)

num_frame = 0
for video in videos_list:
    print(video)
    cap = cv2.VideoCapture(videos_path + '/' + video)

    file_path, file_extension = os.path.splitext(video)
    file_name = os.path.basename(file_path)
    frame_num = 0

    # get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get video's fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get total seconds of video
    total_seconds = int(total_frames / fps)

    # divide total frames by total seconds to get one frame per second
    # my_frame = int(total_frames / total_seconds)

    # or set the jump of frames = fps, so that I get one frame every fps frames (one frame per second)
    skip_frame = int(fps)

    # for every second of the video, I take one frame every skip_frame
    for i in range(1, total_seconds + 2):
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

        if os.path.exists(videos_path + '\\' + label_list[my_frame-1]):
            cv2.imwrite(
                f'{output_path}/{file_name}_{my_frame}.png', frame)

            shutil.copy(videos_path + '\\' + label_list[my_frame-1], output_path)
        else:
            continue
    num_frame += total_frames
    label_list = label_list[num_frame:]

    # cap.release()
    cv2.destroyAllWindows()