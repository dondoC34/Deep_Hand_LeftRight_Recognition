from tf_models import leNet
from Data_Pipeline import data_generator
import os
import pandas as pd

if __name__ == "__main__":
    conv_layers = [(50, 4, 4), (100, 3, 3), (150, 3, 3), (200, 3, 3), (300, 2, 2)]
    dense_layers = [400, 500, 600, 700]
    PIPELINE_BATCH_TEST = 64
    TEST_FOLDER = "../ssd/Test_Frames/"  # FOLDER ON GCP, REPLACE IF NECESSARY
    NUM_OF_TEST_IMAGES = len(os.listdir(TEST_FOLDER + "Right/")) + len(os.listdir(TEST_FOLDER + "Left/"))
    streaming_pipeline_test = data_generator(input_folder=TEST_FOLDER,
                                             data_augmentation=None,
                                             batch_size=PIPELINE_BATCH_TEST,
                                             rescale=(144, 256),
                                             normalization_factor=1/255,
                                             shuffle=False,
                                             mode="test")

    model = leNet(conv_layers=conv_layers,
                  dense_layers=dense_layers,
                  interpose_pooling_layers=True,
                  interpose_dropout=False,
                  input_shape=(256, 144, 1))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    model.load_weights("Model_Weights/We_leNet_esLoss_4_dense")

    predictions = model.predict(x=streaming_pipeline_test,
                                steps=int(NUM_OF_TEST_IMAGES / PIPELINE_BATCH_TEST))

    predictions_nr = len(predictions)
    labels = []
    for i in range(predictions_nr):
        if i % 2 == 0:
            labels.append(0)
        else:
            labels.append(1)
    labeled_predictions = list(zip(predictions, labels))
    
    prediction_frame = pd.DataFrame(labeled_predictions, columns=["Prediction", "Class"])
    prediction_frame.to_csv("Models_Predictions/loss_monitor_model_predictions_dense4.csv", index=False)

    

