import tensorflow as tf
from Data_Pipeline import data_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def leNet(conv_layers, dense_layers, interpose_pooling_layers=False, input_shape=(256, 144, 1)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=conv_layers[0][0],
                                     kernel_size=(conv_layers[0][1], conv_layers[0][2]),
                                     input_shape=input_shape))
    if interpose_pooling_layers:
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    for conv_layer in conv_layers[1:]:
        model.add(tf.keras.layers.Conv2D(filters=conv_layer[0],
                                         kernel_size=(conv_layer[1], conv_layer[2])))
        if interpose_pooling_layers:
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    for dense_layer in dense_layers:
        model.add(tf.keras.layers.Dense(dense_layer, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    tf.get_logger().setLevel('ERROR')
    conv_layers = [(20, 3, 3), (50, 3, 3), (75, 3, 3), (100, 3, 3), (200, 2, 2)]
    dense_layers = [400, 300, 200, 100]
    PIPELINE_BATCH = 64
    PIPELINE_BATCH_TEST = 250
    TRAIN_FOLDER = "Data/Training_Frames/"
    TEST_FOLDER = "Data/Test_Frames/"
    NUM_OF_TRAIN_IMAGES = len(os.listdir(TRAIN_FOLDER + "Right/")) + len(os.listdir(TRAIN_FOLDER + "Left/"))
    NUM_OF_TEST_IMAGES = len(os.listdir(TEST_FOLDER + "Right/")) + len(os.listdir(TEST_FOLDER + "Left/"))
    EPOCHS = NUM_OF_TRAIN_IMAGES / (PIPELINE_BATCH * 145)
    aug = ImageDataGenerator(width_shift_range=0.04,
                             height_shift_range=0.04,
                             zoom_range=0.2,
                             horizontal_flip=False,
                             rescale=1/255,
                             rotation_range=0)

    model = leNet(conv_layers=conv_layers,
                  dense_layers=dense_layers,
                  interpose_pooling_layers=True)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])

    streaming_pipeline_train = data_generator(input_folder=TRAIN_FOLDER,
                                              data_aumentation=None,
                                              batch_size=PIPELINE_BATCH,
                                              rescale=(144, 256),
                                              normalization_factor=1/255)
    streaming_pipeline_test = data_generator(input_folder=TEST_FOLDER,
                                             data_aumentation=None,
                                             batch_size=PIPELINE_BATCH_TEST,
                                             rescale=(144, 256),
                                             normalization_factor=1/255)
    streaming_pipeline_test_pred = data_generator(input_folder=TEST_FOLDER,
                                                  data_aumentation=None,
                                                  batch_size=PIPELINE_BATCH,
                                                  rescale=(144, 256),
                                                  normalization_factor=1 / 255)

    history = model.fit(x=streaming_pipeline_train,
                        steps_per_epoch=145,
                        epochs=int(EPOCHS),
                        validation_data=streaming_pipeline_test,
                        validation_steps=int(NUM_OF_TEST_IMAGES / PIPELINE_BATCH_TEST))

    history_dict = history.history
    loss = history_dict["loss"]
    acc = history_dict["acc"]
    val_loss = history_dict["val_loss"]
    val_acc = history_dict["val_acc"]
    frame_list = []

    for i in range(int(EPOCHS)):
        frame_list.append([x[i] for x in [loss, acc, val_loss, val_acc]])
    frame = pd.DataFrame(frame_list, columns=["loss", "acc", "val-loss", "val-acc"])
    frame.to_csv("History.csv")

    predictions = model.predict(x=streaming_pipeline_test_pred, steps=2)
    print(predictions)

