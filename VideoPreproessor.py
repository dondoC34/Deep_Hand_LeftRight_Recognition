import cv2
from tqdm import tqdm
import os
import numpy as np


def extract_frames_from_video(videopath, output_folder, rotate_90_clockwise=False, use_gray_scale=False,
                              resize_dims=None, label=None, shuffle=True):
    vid = cv2.VideoCapture(videopath)
    index = len(os.listdir(output_folder))
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            if rotate_90_clockwise:
                frame = cv2.rotate(frame, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            if use_gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if resize_dims is not None:
                frame = cv2.resize(frame, resize_dims)

            if shuffle:
                frame_name = str(np.random.uniform(0, 100000))
            else:
                frame_name = str(index)
                index += 1
            if label is not None:
                frame_name = frame_name + "_" + str(label)

            cv2.imwrite(os.path.join(output_folder, frame_name) + ".jpg", frame)
        else:
            break
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    for video in tqdm(os.listdir("Data/Raw_Videos/Test_Videos/")):
        hand_type = video.split("_")[0]
        extract_frames_from_video(videopath=os.path.join("Data/Raw_Videos/Test_Videos", video),
                                  output_folder="Data/Test_Frames/" + hand_type + "/",
                                  rotate_90_clockwise=True,
                                  use_gray_scale=True,
                                  resize_dims=(288, 512),
                                  label=hand_type,
                                  shuffle=False)





