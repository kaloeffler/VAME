"""Print the architecture of a model"""
import torch
from vame.model.rnn_vae import RNN_VAE
from datetime import datetime
import os
from vame.util.auxiliary import read_config
from torchinfo import summary


def print_model_architecture(project_path):
    # load the CONFIG FILE from the last trained model
    trained_models = [
        (datetime.strptime(element, "%m-%d-%Y-%H-%M"), element)
        for element in os.listdir(os.path.join(project_path, "model"))
    ]
    # sort by time step
    trained_models.sort(key=lambda x: x[0])
    latest_model = trained_models[-1][-1]

    config_file = os.path.join(project_path, "model", latest_model, "config.yaml")
    config = read_config(config_file)


    ZDIMS = config["zdims"]
    NUM_FEATURES = config["num_features"]
    NUM_FEATURES = NUM_FEATURES - 2
    TEMPORAL_WINDOW = config["time_window"] * 2
    FUTURE_DECODER = config["prediction_decoder"]
    FUTURE_STEPS = config["prediction_steps"]

    # RNN
    hidden_size_layer_1 = config["hidden_size_layer_1"]
    hidden_size_layer_2 = config["hidden_size_layer_2"]
    hidden_size_rec = config["hidden_size_rec"]
    hidden_size_pred = config["hidden_size_pred"]
    dropout_encoder = config["dropout_encoder"]
    dropout_rec = config["dropout_rec"]
    dropout_pred = config["dropout_pred"]
    noise = config["noise"]
    scheduler_step_size = config["scheduler_step_size"]
    softplus = config["softplus"]


    model = RNN_VAE(
                TEMPORAL_WINDOW,
                ZDIMS,
                NUM_FEATURES,
                FUTURE_DECODER,
                FUTURE_STEPS,
                hidden_size_layer_1,
                hidden_size_layer_2,
                hidden_size_rec,
                hidden_size_pred,
                dropout_encoder,
                dropout_rec,
                dropout_pred,
                softplus,
            ).to()


    summary(model, input_size=(1,TEMPORAL_WINDOW, NUM_FEATURES))


if __name__ == "__main__":
    PROJECT_PATH = "/home/katharina/vame_approach/themis_tail_belly_align"
    print_model_architecture(PROJECT_PATH)