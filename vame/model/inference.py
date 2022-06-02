"""Apply a trained model on another landmark file"""
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


def embedd_data(data_files, cfg, model, legacy, train_data_path, batch_size=256):
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
                # for i in tqdm.tqdm(range(10000)):
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


if __name__ == "__main__":
    PROJECT_PATH = "/home/katharina/vame_approach/themis_tail_belly_align"
    SAVE_DATA_PATH = os.path.join(PROJECT_PATH, "inference")
    # per landmark file there will be a subdir in inference named as the landmark and will store the csv and the video data?

    # video: rhodent
    # caution: the pkl files just contain the a fraction (the first N frames) of the total landmarks
    # 0057: H01 -> the landmarks the model in "themis_tail_belly_align" was trained on
    # 0051: K8
    # 0089: H06
    ## 0053: (full dataset!)
    PKL_FILE = "/media/Themis/Data/Models/3843S2B10Gaussians/analyses/landmarks_0089_3843S2B10Gaussians_E149_confidece.pkl"

    pkl_file_dir = os.path.dirname(PKL_FILE)
    # load the CONFIG FILE from the last trained model
    trained_models = [
        (datetime.strptime(element, "%m-%d-%Y-%H-%M"), element)
        for element in os.listdir(os.path.join(PROJECT_PATH, "model"))
    ]
    # sort by time step
    trained_models.sort(key=lambda x: x[0])
    latest_model = trained_models[-1][-1]

    config_file = os.path.join(PROJECT_PATH, "model", latest_model, "config.yaml")

    # prepare data pkl -> csv -> data as model input
    pkl_file_name = os.path.basename(PKL_FILE).split(".")[0]
    landmark_file_name = "_".join(pkl_file_name.split("_")[:-1])
    landmark_file = os.path.join(pkl_file_dir, landmark_file_name + ".csv")
    pose_alignment_idx = np.load(
        os.path.join(PROJECT_PATH, "data", "train", "pose_alignment_idx.npy")
    ).to_list()

    if True:

        if not os.path.exists(landmark_file):
            print(f"No file named {landmark_file}")

        landmark_names = pd.read_csv(landmark_file, header=[0, 1]).columns
        landmark_df = pickle_dlc_to_df(PKL_FILE, landmark_names, return_conf_df=False)
        save_landmark_file = os.path.join(
            SAVE_DATA_PATH, "landmarks", pkl_file_name + ".csv"
        )
        if not os.path.exists(os.path.dirname(save_landmark_file)):
            os.makedirs(os.path.dirname(save_landmark_file))
        landmark_df.to_csv(save_landmark_file)

        align_inference_data(
            save_landmark_file,
            config_file,
            alignment_idx=pose_alignment_idx,
            save_dir=SAVE_DATA_PATH,
        )

        aligned_data_file = os.path.join(
            SAVE_DATA_PATH, "data", pkl_file_name, pkl_file_name + "-PE-seq.npy",
        )
        train_data_dir = os.path.join(PROJECT_PATH, "data", "train")
        preprocess_inference_data(aligned_data_file, config_file, train_data_dir)

    # TODO: collect automatically from the data folder
    dd = os.path.join(
        SAVE_DATA_PATH, "data", pkl_file_name, pkl_file_name + "-PE-seq-clean.npy",
    )
    inference_data_files = [dd]

    res_path = os.path.join(SAVE_DATA_PATH, "results")
    #  load model and predict embeddings for the aligned and preprocessed data files
    inference(inference_data_files, config_file, train_data_dir, res_path)

    # todo: evaluate the embeddings and compare to the orig found clusters?
    # e.g. fit on "train"data - cluster using the fcm fittet on train then vis clip matrices
    # sampled from the "same" cluster

    # howmuch do the clusters found in train vs found by fitting fcm again on vectors from another dataset
    # look alike? - compare by finding the overlap of samples clustered to cluster N_x / n_y
