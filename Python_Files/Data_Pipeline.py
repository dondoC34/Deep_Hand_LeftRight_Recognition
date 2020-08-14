import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import random


def data_generator(input_folder, batch_size, normalization_factor=None, mode="train", data_augmentation=None,
                   rescale=None, verbose=0, shuffle=True):
    
    right_images = os.listdir(os.path.join(input_folder, "Right/"))
    left_images = os.listdir(os.path.join(input_folder, "Left/"))

    if shuffle:
        random.shuffle(right_images)
        random.shuffle(left_images)

    num_of_right_images = len(right_images)
    num_of_left_images = len(left_images)
    right_index = 0
    left_index = 0
    right = True
    print_first = False

    while True:
        images = []
        labels = []

        while len(images) < batch_size:
            # TAKE WITH EQUAL PROBABILITY AN IMAGE FROM THE LEFT FOLDER OR THE RIGHT FOLDER
            # THIS AVOID THE FACT THAT WE TRAIN THE CNN WITH ALL THE LEFT HANDS BEFORE THE RIGHT HANDS
            if right:
                right = False
                image = cv2.imread(os.path.join(input_folder, "Right/" + right_images[right_index]),
                                   cv2.IMREAD_GRAYSCALE)
                labels.append(0)
                if (not print_first) and (verbose == 1):
                    print(right_images[right_index])
                    print_first = True

                right_index += 1
                if right_index + 1 == num_of_right_images:  # IF WE HAVE GENERATED ALL THE IMAGES, RESTART
                    right_index = 0
            else:
                right = True
                image = cv2.imread(os.path.join(input_folder, "Left/" + left_images[left_index]), cv2.IMREAD_GRAYSCALE)
                if (not print_first) and (verbose == 1):
                    print(left_images[left_index])
                    print_first = True
                left_index += 1
                if left_index + 1 == num_of_left_images:
                    left_index = 0
                labels.append(1)

            if rescale is not None:
                image = cv2.resize(image, rescale)
            images.append(image)

        if (mode == "train") and (data_augmentation is not None):
            images = np.array(images)
            images = images.reshape(batch_size, 256, 144, 1)
            images, labels = next(data_augmentation.flow(np.array(images), labels, batch_size=batch_size))
            yield images, labels
        elif mode == "eval":
            images = np.array(images)
            images = images.reshape(batch_size, 256, 144, 1)
            if normalization_factor is not None:
                yield normalization_factor * images, labels
            else:
                yield images, labels
        elif mode == "test":
            images = np.array(images)
            images = images.reshape(batch_size, 256, 144, 1)
            if normalization_factor is not None:
                yield normalization_factor * images
            else:
                yield images


if __name__ == "__main__":
    frame = pd.read_csv("Models_History/History.csv")
    plt.plot(frame["loss"])
    plt.plot(frame["val-loss"])
    plt.show()


