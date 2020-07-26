import tensorflow as tf
from Data_Pipeline import data_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def leNet(conv_layers, dense_layers, interpose_pooling_layers=False, input_shape=(256, 144, 3)):
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
    conv_layers = [(50, 4, 4), (100, 3, 3), (150, 3, 3), (200, 3, 3), (300, 2, 2)]
    dense_layers = [400, 500, 600, 700]
    PIPELINE_BATCH = 64
    PIPELINE_BATCH_TEST = 100
    TRAIN_FOLDER = "../ssd/Training_Frames/"
    TEST_FOLDER = "../ssd/Test_Frames/"
    NUM_OF_TRAIN_IMAGES = len(os.listdir(TRAIN_FOLDER + "Right/")) + len(os.listdir(TRAIN_FOLDER + "Left/"))
    NUM_OF_TEST_IMAGES = len(os.listdir(TEST_FOLDER + "Right/")) + len(os.listdir(TEST_FOLDER + "Left/"))
    EPOCHS = 200
    es = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=10, restore_best_weights=True)
    aug = ImageDataGenerator(width_shift_range=0.05,
                             height_shift_range=0.05,
                             zoom_range=0.25,
                             horizontal_flip=False,
                             rescale=1/255,
                             rotation_range=0)

    model = leNet(conv_layers=conv_layers,
                  dense_layers=dense_layers,
                  interpose_pooling_layers=True)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])

    streaming_pipeline_train = data_generator(input_folder=TRAIN_FOLDER,
                                              data_aumentation=aug,
                                              batch_size=PIPELINE_BATCH,
                                              rescale=(144, 256))

    streaming_pipeline_test = data_generator(input_folder=TEST_FOLDER,
                                             data_aumentation=None,
                                             batch_size=PIPELINE_BATCH_TEST,
                                             rescale=(144, 256),
                                             normalization_factor=1/255)

    history = model.fit(x=streaming_pipeline_train,
                        steps_per_epoch=100,
                        epochs=int(EPOCHS),
                        validation_data=streaming_pipeline_test,
                        validation_steps=int(NUM_OF_TEST_IMAGES / PIPELINE_BATCH_TEST),
                        callbacks=[es])

    history_dict = history.history
    loss = history_dict["loss"]
    acc = history_dict["acc"]
    val_loss = history_dict["val_loss"]
    val_acc = history_dict["val_acc"]
    frame_list = []

    for i in range(len(loss)):
        frame_list.append([x[i] for x in [loss, acc, val_loss, val_acc]])
    frame = pd.DataFrame(frame_list, columns=["loss", "acc", "val-loss", "val-acc"])
    frame.to_csv("History_colored.csv")

    model.save_weights("best_model_color_weights")

