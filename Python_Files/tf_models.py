import tensorflow as tf
from Python_Files.Data_Pipeline import data_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def leNet(conv_layers, dense_layers, interpose_pooling_layers=False, interpose_dropout=False, input_shape=(256, 144, 1)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=conv_layers[0][0],
                                     kernel_size=(conv_layers[0][1], conv_layers[0][2]),
                                     input_shape=input_shape,
                                     activation="relu"))
    if interpose_pooling_layers:
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    if interpose_dropout:
        model.add(tf.keras.layers.Dropout(rate=0.2))

    for conv_layer in conv_layers[1:]:
        model.add(tf.keras.layers.Conv2D(filters=conv_layer[0],
                                         kernel_size=(conv_layer[1], conv_layer[2]),
                                         activation="relu"))
        if interpose_pooling_layers:
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    for dense_layer in dense_layers:
        model.add(tf.keras.layers.Dense(dense_layer, activation="relu"))

    # output layer for binary classification
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


def shapedNet(first_step_conv_layers, second_step_conv_layers, first_step_dense_layers, final_dense_layers,
              interpose_pooling_layers, input_shape=(256, 144, 1)):

    model_input = tf.keras.Input(shape=input_shape)
    x_input = tf.keras.layers.Conv2D(filters=first_step_conv_layers[0][0],
                                     kernel_size=(first_step_conv_layers[0][1],
                                                  first_step_conv_layers[0][2]))(model_input)
    if interpose_pooling_layers:
        x_input = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x_input)

    for conv_layer in first_step_conv_layers[1:]:
        x_input = tf.keras.layers.Conv2D(filters=conv_layer[0],
                                         kernel_size=(conv_layer[1], conv_layer[2]))(x_input)
        if interpose_pooling_layers:
            x_input = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x_input)

    x_first_dense = tf.keras.layers.Flatten()(x_input)
    x_first_dense = tf.keras.layers.Dense(first_step_dense_layers[0], activation="relu")(x_first_dense)
    x_second_step_conv = tf.keras.layers.Conv2D(filters=second_step_conv_layers[0][0],
                                                kernel_size=(second_step_conv_layers[0][1],
                                                             second_step_conv_layers[0][2]))(x_input)
    if interpose_pooling_layers:
        x_second_step_conv = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x_second_step_conv)

    for conv_layer in second_step_conv_layers[1:]:
        x_second_step_conv = tf.keras.layers.Conv2D(filters=conv_layer[0],
                                                    kernel_size=(conv_layer[1], conv_layer[2]))(x_second_step_conv)
        if interpose_pooling_layers and (second_step_conv_layers.index(conv_layer) < len(second_step_conv_layers) - 1):
            x_second_step_conv = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x_second_step_conv)

    x_second_step_conv = tf.keras.layers.Flatten()(x_second_step_conv)

    for dense_layer in first_step_dense_layers[1:]:
        x_first_dense = tf.keras.layers.Dense(dense_layer, activation="relu")(x_first_dense)

    # MERGE THE TWO SIDES
    x_final_dense = tf.keras.layers.Concatenate()([x_first_dense, x_second_step_conv])
    for dense_layer in final_dense_layers:
        x_final_dense = tf.keras.layers.Dense(dense_layer, activation="relu")(x_final_dense)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x_final_dense)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    return model


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    tf.get_logger().setLevel('ERROR')
    first_conv_layers = [(32, 4, 4), (64, 3, 3)]
    second_conv_layers = [(128, 2, 2), (256, 2, 2)]
    first_dense_layers = [500, 600, 700]
    second_dense_layers = [500, 600, 700]
    conv_layers = [[100, 4, 4], [150, 3, 3], [200, 3, 3], [250, 3, 3], [350, 2, 2]]
    dense_layers = [400, 500, 600, 700]
    PIPELINE_BATCH = 64
    PIPELINE_BATCH_TEST = 100
    TRAIN_FOLDER = "../ssd/Training_Frames/"
    TEST_FOLDER = "../ssd/Test_Frames/"
    NUM_OF_TRAIN_IMAGES = len(os.listdir(TRAIN_FOLDER + "Right/")) + len(os.listdir(TRAIN_FOLDER + "Left/"))
    NUM_OF_TEST_IMAGES = len(os.listdir(TEST_FOLDER + "Right/")) + len(os.listdir(TEST_FOLDER + "Left/"))
    EPOCHS = 200
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    aug = ImageDataGenerator(width_shift_range=0.05,
                             height_shift_range=0.05,
                             zoom_range=0.25,
                             horizontal_flip=False,
                             rescale=1 / 255,
                             rotation_range=0)
    # model = shapedNet(first_step_conv_layers=first_conv_layers,
    #                   second_step_conv_layers=second_conv_layers,
    #                   first_step_dense_layers=first_dense_layers,
    #                   final_dense_layers=second_dense_layers,
    #                   interpose_pooling_layers=True,
    #                   input_shape=(256, 144, 1))

    for _ in range(4):
        model = leNet(conv_layers=conv_layers,
                      dense_layers=dense_layers,
                      interpose_pooling_layers=True,
                      input_shape=(256, 144, 1))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        streaming_pipeline_train = data_generator(input_folder=TRAIN_FOLDER,
                                                  data_augmentation=aug,
                                                  batch_size=PIPELINE_BATCH,
                                                  rescale=(144, 256),
                                                  mode="train",
                                                  shuffle=True)
        streaming_pipeline_test = data_generator(input_folder=TEST_FOLDER,
                                                 data_augmentation=None,
                                                 batch_size=PIPELINE_BATCH_TEST,
                                                 rescale=(144, 256),
                                                 normalization_factor=1 / 255,
                                                 mode="eval",
                                                 shuffle=False)
        history = model.fit(x=streaming_pipeline_train,
                            steps_per_epoch=100,
                            epochs=EPOCHS,
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
        frame.to_csv("Models_History/Hist_leNet_esLoss_4_dense_fm_nr_{}.csv".format(conv_layers[0][0]))

        for k in range(len(conv_layers)):
            conv_layers[k][0] += 50
    # model.save_weights("Model_Weights/We_leNet_esLoss_0_dense")


