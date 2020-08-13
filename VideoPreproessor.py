import cv2
from tqdm import tqdm
import os
import numpy as np


def extract_frames_from_video(videopath, output_folder, rotate_90_clockwise=False, use_gray_scale=False,
                              resize_dims=None, label=None, shuffle=True, acquisition_frame_rate=1):
    vid = cv2.VideoCapture(videopath)
    index = len(os.listdir(output_folder))
    frame_rate_count = 0
    acquire = True
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            if acquire:
                if rotate_90_clockwise:
                    frame = cv2.rotate(frame, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                if use_gray_scale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if resize_dims is not None:
                    frame = cv2.resize(frame, resize_dims)

                if shuffle:
                    frame_name = str(np.random.uniform(0, 1000000))
                else:
                    frame_name = str(index)
                    index += 1
                if label is not None:
                    frame_name = frame_name + "_" + str(label)

                if frame.shape[0] < frame.shape[1]:
                    print("orientation error, expecting vertical orientation. Aborting.")
                    exit(1)

                cv2.imwrite(os.path.join(output_folder, frame_name) + ".jpg", frame)
            frame_rate_count += 1
            if frame_rate_count < acquisition_frame_rate:
                acquire = False
            else:
                acquire = True
                frame_rate_count = 0
        else:
            break
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    videos_folder = "../ssd/05-08-20/Test"
    
    for video in tqdm(os.listdir(videos_folder + "/")):
        hand_type = video.split("_")[0]
        extract_frames_from_video(videopath=os.path.join(videos_folder, video),
                                  output_folder="Real-Dataset/Training_Frames/" + hand_type + "/",
                                  rotate_90_clockwise=True,
                                  use_gray_scale=True,
                                  resize_dims=(144, 256),
                                  label=hand_type,
                                  shuffle=False,
                                  acquisition_frame_rate=1)





