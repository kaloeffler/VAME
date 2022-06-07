"""Inference utilities to apply a trained model on a landmark file and predict latent vectors."""
import os
from datetime import datetime
import pandas as pd
from vame.util.prep_themis_data import pickle_dlc_to_df
from vame.util import read_config
from vame.util.align_egocentrical_themis import alignment
import numpy as np
from scipy.stats import iqr
from vame.model.create_training import interpol
from scipy.signal import savgol_filter
from vame.analysis.pose_segmentation import load_model
import torch
import tqdm


def align_inference_data(
    landmark_file: str, config_file: str, alignment_idx: list, save_dir: str
):
    """Prepare landmark data which is in csv format to a np.array of aligned landmarks.

    Args:
        landmark_file (str): path to the landmark csv file that shall be processed
        config_file (str): path to the config file
        alignment_idx (list): landmark indices to align the landmarks to
        time_idx_to_delete (list): time series from the landmark data that will be removed, as in the training data
        save_dir (str): dir in which the aligned data will be saved
    """
    config = read_config(config_file)
    project_dir = os.path.dirname(os.path.dirname(landmark_file))
    landmark_file_name = os.path.basename(landmark_file).split(".")[0]

    aligned_data, _ = alignment(
        "",
        project_dir,
        landmark_file_name,
        alignment_idx,
        (300, 300),
        config["pose_confidence"],
        use_video=False,
        check_video=False,
    )
    if not os.path.exists(os.path.join(save_dir, "data", landmark_file_name)):
        os.makedirs(os.path.join(save_dir, "data", landmark_file_name))
    np.save(
        os.path.join(
            save_dir, "data", landmark_file_name, landmark_file_name + "-PE-seq.npy"
        ),
        aligned_data,
    )


def preprocess_inference_data(
    aligned_data_file: str, config_file: str, train_data_path: str
):
    """Preprocess the aligned landmark data.

    Args:
        aligned_data_file (str): path to the aligned landmark data
        config_file (str): path to the config file
        train_data_path (str): path to where the training data was stored
    """
    aligned_data = np.load(aligned_data_file)
    config = read_config(config_file)

    time_idx_to_remove = np.load(
        os.path.join(train_data_path, "timeseries_idx_deleted.npy")
    )

    # apply mean and std from train data
    x_mean = np.load(os.path.join(train_data_path, "normalize_mean.npy"))

    x_std = np.load(os.path.join(train_data_path, "normalize_std.npy"))

    x_z = (aligned_data.T - x_mean) / x_std

    if config["robust"] == True:
        iqr_val = iqr(x_z)
        print(
            "IQR value: %.2f, IQR cutoff: %.2f"
            % (iqr_val, config["iqr_factor"] * iqr_val)
        )
        for i in range(x_z.shape[0]):
            for marker in range(x_z.shape[1]):
                if x_z[i, marker] > config["iqr_factor"] * iqr_val:
                    x_z[i, marker] = np.nan

                elif x_z[i, marker] < -config["iqr_factor"] * iqr_val:
                    x_z[i, marker] = np.nan

            x_z[i, :] = interpol(x_z[i, :])

    sorted(time_idx_to_remove)
    for idx in time_idx_to_remove:
        x_z = np.delete(x_z, idx, 1)

    x_z = x_z.T
    if config["savgol_filter"]:
        x = savgol_filter(x_z, config["savgol_length"], config["savgol_order"])
    else:
        x = x

    file_name, ending = aligned_data_file.split(".")
    np.save(file_name + "-clean." + ending, x)


def inference(
    inference_data_files: list,
    config_file: str,
    train_data_path: str,
    save_res_path: str,
):
    """Predict from aligned, preprocessed landmark data files latent embeddings.

    Args:
        inference_data_file (list): list of paths to a preprocessed, aligned data files to predict latent embeddings from
        config_file (str): path to the config file
        train_data_path (str): path where the training  data is stored
        save_res_path (str): path where the results will be stored
    """
    config = read_config(config_file)
    model_name = config["model_name"]

    model = load_model(config, model_name, config["legacy"])
    latent_vectors_all = embedd_data(
        inference_data_files, config, model, config["legacy"], train_data_path
    )

    # todo: split latent vectors to sequences and save separately
    for i_seq, latent_vectors_seq in enumerate(latent_vectors_all):

        name_inference_file = os.path.basename(inference_data_files[i_seq]).split(
            "-PE-seq-clean.npy"
        )[0]
        if not os.path.exists(os.path.join(save_res_path, config["time_stamp"])):
            os.makedirs(os.path.join(save_res_path, config["time_stamp"]))
        np.save(
            os.path.join(
                save_res_path,
                config["time_stamp"],
                "latent_vectors_" + name_inference_file + ".npy",
            ),
            latent_vectors_seq,
        )


def embedd_data(
    data_files: list,
    cfg: dict,
    model: torch.nn.Module,
    legacy: bool,
    train_data_path: str,
    batch_size: int = 256,
):
    """Predict latent vectors for landmark time series.

    Args:
        data_files (list): list of path (.npy files)s to preprocessed, aligned landmark data
        cfg (dict): configuration of the project
        model (torch.nn.Module): the trained VAE model
        legacy (bool): if legacy is true the full number of features is used, otherwise n_features -2
                     (the two time series with the lowest std in the train data are removed)
        train_data_path (str): path to the training data
        batch_size (int, optional): batch size to process batches of time series simultaneously. Defaults to 256.
    """
    temp_win = cfg["time_window"]
    num_features = cfg["num_features"]
    if legacy == False:
        num_features = num_features - 2

    latent_vector_files = []

    # load mean and std from training data and normalize the data as
    # in training: see SEQUENCE_DATASET in vame/model/dataloader.py
    x_mean = np.load(os.path.join(train_data_path, "seq_mean.npy"))
    x_std = np.load(os.path.join(train_data_path, "seq_std.npy"))

    for file in data_files:
        print("Embedd latent vectors for file %s" % file)
        data = np.load(file)
        latent_vector_list = []
        with torch.no_grad():
            data_normalized = (data - x_mean) / x_std
            for i in tqdm.tqdm(range(0, data.shape[1] - temp_win, batch_size)):
                temp_win_ids = np.arange(temp_win)
                time_ids = np.arange(i, min(batch_size + i, data.shape[1] - temp_win))
                data_ids = time_ids.reshape(-1, 1) + temp_win_ids.reshape(1, -1)
                # shape: (num_features, batchsize, temp_win)
                data_sample_np = data_normalized[:, data_ids]
                # shape: (batchsize, temp_win, num_features)
                data_sample_np = np.transpose(data_sample_np, axes=(1, 2, 0))
                h_n = model.encoder(
                    torch.from_numpy(data_sample_np).type("torch.FloatTensor").cuda()
                )
                _, mu, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())

        latent_vector = np.concatenate(latent_vector_list, axis=0)
        latent_vector_files.append(latent_vector)

    return latent_vector_files
