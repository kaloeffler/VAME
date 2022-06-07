"""Adapt the new.py file from the orig. VAME project for the themis data"""
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from vame.util.prep_themis_data import get_video_metadata, pickle_dlc_to_df
from vame.util import auxiliary


def init_new_project(
    project_path: str, video_root: str, landmark_root: str, select_video_ids: list = []
):
    """Create a new project for VAME. This will create the required folders as well as create csv landmark
    files from pickle files that contain the landmarks with confidence scores.

    Args:
        project_path (str): path where the new project will be created
        video_root (str): path to the video root of the THEMIS project
        landmark_root (str): path to the landmark.pkl files to use (which contain the landmark positions and the confidence scores)
        select_video_ids (list, optional): . Defaults to [] - all landmarks files will be used in training.

    Returns:
        str: path to the config.yaml file
    """

    if os.path.exists(project_path):
        raise AssertionError(f"Project dir {project_path} already exists!")
    project_path = Path(project_path)
    landmark_path = project_path / "landmarks"
    data_path = project_path / "data"
    results_path = project_path / "results"
    model_path = project_path / "model"
    inference_path = project_path / "inference"

    for p in [landmark_path, data_path, results_path, model_path, inference_path]:
        p.mkdir(parents=True)
        print('Created "{}"'.format(p))

    # generate csv files and save location of corresponding videos
    save_video_and_landmark_dfs(video_root, landmark_root, project_path, landmark_path)

    # create for each landmark file a subdir in data and in results
    landmark_file_names = [file.split(".")[0] for file in os.listdir(landmark_path)]
    if select_video_ids:
        landmark_file_names = [
            file
            for file in landmark_file_names
            if file.split("_")[1] in select_video_ids
        ]
    for lm_name in landmark_file_names:
        (results_path / lm_name).mkdir(parents=True, exist_ok=True)
        (data_path / lm_name).mkdir(parents=True, exist_ok=True)

    cfg_file, ruamelFile = auxiliary.create_config_template()
    cfg_file

    cfg_file["Project"] = Path(project_path).name
    cfg_file["project_path"] = str(project_path) + "/"
    # timestamp will be written when a model is created at training
    cfg_file["time_stamp"] = "None"
    cfg_file["test_fraction"] = 0.1
    # since we have more landmark predictions ( trained with different approaches)
    #  than videos - use the names of the landmark predictions instead
    cfg_file["video_sets"] = landmark_file_names
    cfg_file["all_data"] = "yes"
    cfg_file["load_data"] = "-PE-seq-clean"
    cfg_file["anneal_function"] = "linear"
    cfg_file["batch_size"] = 256
    cfg_file["max_epochs"] = 500
    cfg_file["transition_function"] = "GRU"
    cfg_file["beta"] = 1
    cfg_file["zdims"] = 30
    cfg_file["learning_rate"] = 5e-4
    # FIXME: the videos in the VAME paper have frame rate 25 (see video example), we use 120fps
    #  so adapting the window size to a comparable total time span here
    cfg_file["time_window"] = int(30 * 120 / 25)
    cfg_file["prediction_decoder"] = 1
    # FIXME: the videos in the VAME paper have frame rate 25 (see video example on the webside), we use 120fps
    #  so adapting the window size to a comparable total time span here
    cfg_file["prediction_steps"] = int(15 * 120 / 25)
    cfg_file["model_convergence"] = 50
    cfg_file["model_snapshot"] = 50
    cfg_file["num_features"] = 34  # number of landmarks * 2
    cfg_file["savgol_filter"] = True
    cfg_file["savgol_length"] = 5
    cfg_file["savgol_order"] = 2
    cfg_file["hidden_size_layer_1"] = 256
    cfg_file["hidden_size_layer_2"] = 256
    cfg_file["dropout_encoder"] = 0
    cfg_file["hidden_size_rec"] = 256
    cfg_file["dropout_rec"] = 0
    cfg_file["hidden_size_pred"] = 256
    cfg_file["dropout_pred"] = 0
    cfg_file["kl_start"] = 2
    cfg_file["annealtime"] = 4
    cfg_file["mse_reconstruction_reduction"] = "sum"
    cfg_file["mse_prediction_reduction"] = "sum"
    cfg_file["kmeans_loss"] = cfg_file["zdims"]
    cfg_file["kmeans_lambda"] = 0.1
    cfg_file["scheduler"] = 1
    cfg_file["length_of_motif_video"] = 1000
    cfg_file["noise"] = False
    cfg_file["scheduler_step_size"] = 100
    cfg_file["legacy"] = False
    cfg_file["individual_parameterization"] = False
    cfg_file["random_state_kmeans"] = 42
    cfg_file["n_init_kmeans"] = 15
    cfg_file["model_name"] = "VAME"
    cfg_file["n_cluster"] = 15
    cfg_file["pretrained_weights"] = False
    # if pretrained_weights = True provide the full path to the model
    # folder (../.../model) that contains the model weights you want to use here
    cfg_file["pretrained_model"] = "None"
    cfg_file["min_dist"] = 0.1
    cfg_file["n_neighbors"] = 200
    cfg_file["random_state"] = 42
    cfg_file["num_points"] = 30000
    cfg_file["scheduler_gamma"] = 0.2
    cfg_file["softplus"] = False
    # TODO: set to something resonable concerning the confidence scores of the
    # landmark files -> use the lm_conf_quantiles.py in analysis_scripts to check
    # how the confidence scores are distributed
    cfg_file["pose_confidence"] = 0.5
    cfg_file["iqr_factor"] = 4
    cfg_file["robust"] = True
    cfg_file["beta_norm"] = False
    cfg_file["n_layers"] = 1
    cfg_file["axis"] = None

    projconfigfile = os.path.join(str(project_path), "config.yaml")
    # Write dictionary to yaml  config file
    auxiliary.write_config(projconfigfile, cfg_file)
    return projconfigfile


def save_video_and_landmark_dfs(
    video_root: str, landmark_root: str, save_dir_video: str, save_dir_landmarks: str
):
    """Create landmark.csv files with a format similar to the DLC (deep lab cut) files: x,y,score for each landmark
    based on the landmark.pkl files which contain the landmark x,y position as well as a confidence score.

    Args:
        video_root (str): path to the root where the rat videos are stored
        landmark_root (str): path to the root where the corresponding landmark.pkl files are stored
        save_dir_video (str): path where to save the information on where the videos are located
        save_dir_landmarks (str): path to where the generated landmarks.csv files will be stored
    """
    pkl_files = [
        os.path.join(landmark_root, element)
        for element in os.listdir(landmark_root)
        if element.endswith(".pkl")
    ]

    video_df = get_video_metadata(video_root, DLC_model="")
    video_df["video_id"] = video_df["vid_file"].apply(lambda x: x.split(".")[0])
    # map videos to pkl files
    video_ids = [
        re.findall(r"\d+", os.path.basename(pkl_file))[0] for pkl_file in pkl_files
    ]
    video_df = video_df[np.isin(video_df["video_id"].values, video_ids)]

    # pkl to dataframe
    for pkl_file in pkl_files:
        dlc_file = os.path.join(
            landmark_root, "_".join(os.path.basename(pkl_file).split("_")[:-1]) + ".csv"
        )
        if not os.path.exists(dlc_file):
            print(f"No file named {dlc_file}")
            continue
        landmark_names = pd.read_csv(dlc_file, header=[0, 1]).columns
        landmark_df = pickle_dlc_to_df(pkl_file, landmark_names, return_conf_df=False)
        landmark_df.to_csv(
            os.path.join(
                save_dir_landmarks, os.path.basename(pkl_file).split(".")[0] + ".csv"
            )
        )
    video_df.to_csv(os.path.join(save_dir_video, "video_info.csv"))
