import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd


def data_generator(input_folder, batch_size, normalization_factor=None, mode="train", data_aumentation=None,
                   rescale=None, verbose=0):
    right_images = os.listdir(os.path.join(input_folder, "Right/"))
    left_images = os.listdir(os.path.join(input_folder, "Left/"))
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
                image = cv2.imread(os.path.join(input_folder, "Left/" + left_images[left_index]),
                                   cv2.IMREAD_GRAYSCALE)
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

        if (mode == "train") and (data_aumentation is not None):
            images = np.array(images)
            images = images.reshape(batch_size, 256, 144, 1)
            images, labels = next(data_aumentation.flow(np.array(images), labels, batch_size=batch_size))
            yield images, labels
        else:
            images = np.array(images)
            images = images.reshape(batch_size, 256, 144, 1)
            if normalization_factor is not None:
                yield normalization_factor * images, labels
            else:
                yield images, labels


if __name__ == "__main__":
    frame = pd.read_csv("History.csv")
    plt.plot(frame["loss"])
    plt.plot(frame["val-loss"])
    plt.show()
    exit(3)

    TEST_FOLDER = "Data/Test_Frames/"
    NUM_OF_TEST_IMAGES = len(os.listdir(TEST_FOLDER + "Right/")) + len(os.listdir(TEST_FOLDER + "Left/"))
    for image, label in data_generator(TEST_FOLDER,
                                       64,
                                       rescale=(144, 256)):
        pass


