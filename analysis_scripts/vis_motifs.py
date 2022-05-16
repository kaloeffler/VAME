"""Visualize the learned motifs"""
import sys

sys.path.insert(0, "/home/katharina/vame_approach/VAME")
import os
from datetime import datetime
from vame.util.auxiliary import read_config
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from vame.analysis.visualize import create_pose_snipplet
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_PATH = "/home/katharina/vame_approach/themis_tail_belly_align"
# multiply with window length to find
min_dist_nn_factor = 2
dist_between_neighbors_factor = 0.5
selected_idx = 100
# load the CONFIG FILE from the last trained model
trained_models = [
    (datetime.strptime(element, "%m-%d-%Y-%H-%M"), element)
    for element in os.listdir(os.path.join(PROJECT_PATH, "model"))
]
# sort by time step
trained_models.sort(key=lambda x: x[0])
latest_model = trained_models[-1][-1]

config_file = os.path.join(PROJECT_PATH, "model", latest_model, "config.yaml")
config = read_config(config_file)
## todo save as mp4 not as avi

# TODO: load predicted labels; latent vectors
for landmark_file in config["video_sets"]:
    data_path = os.path.join(
        PROJECT_PATH,
        "results",
        latest_model,
        landmark_file,
        config["model_name"],
        "kmeans-" + str(config["n_init_kmeans"]),
    )
    motif_labels = np.load(
        os.path.join(
            data_path,
            str(config["n_init_kmeans"]) + "_km_label_" + landmark_file + ".npy",
        )
    )
    latent_vectors = np.load(
        os.path.join(data_path, "latent_vector_" + landmark_file + ".npy")
    )
    # TODO: export the time idx as well so we can get the exmple sequence
    cluster_center = np.load(
        os.path.join(data_path, "cluster_center_" + landmark_file + ".npy")
    )
    # extract landmark data and names
    column_names = pd.read_csv(
        os.path.join(PROJECT_PATH, "landmarks", landmark_file + ".csv"), header=[0, 1],
    ).columns
    landmark_names = [col_name[0] for col_name in column_names if col_name[-1] == "x"]

    landmark_data_file = os.path.join(
        PROJECT_PATH, "data", landmark_file, landmark_file + "-PE-seq.npy"
    )
    landmark_data = np.load(landmark_data_file).T
    # reshape to (N_samples, N_landmarks, 2)
    landmark_data = landmark_data.reshape(landmark_data.shape[0], -1, 2)
    # landmark_data = np.transpose(landmark_data, axes=(0,2,1))

    # min max normalize the data to a fixed grid shape for visualization
    landmark_data = (
        (landmark_data - landmark_data.min())
        / (landmark_data.max() - landmark_data.min())
        * 399
    )

    # calc for each latent vector its N nearest neighbors
    print("Calc neighbors")
    # remove time steps arround selected vector
    window_start = max(
        0, selected_idx - int(config["time_window"] * min_dist_nn_factor)
    )
    window_end = min(
        len(latent_vectors),
        selected_idx + int(config["time_window"] * min_dist_nn_factor),
    )

    selected_data = latent_vectors[selected_idx, :]
    time_points = np.arange(0, latent_vectors.shape[0])
    time_points = np.concatenate(
        [time_points[0:window_start], time_points[window_end:-1]]
    )
    latent_vectors = np.concatenate(
        [latent_vectors[0:window_start], latent_vectors[window_end:-1]]
    )

    neighbors = NearestNeighbors(n_neighbors=4).fit(latent_vectors)
    dist, neighbor_idx = neighbors.kneighbors(selected_data.reshape(1, -1))
    print(neighbor_idx.shape)
    # all dist to selected latent vector
    dist = np.sqrt(np.sum((latent_vectors - selected_data.reshape(1, -1)) ** 2, axis=1))
    dist_temporal = np.sqrt((time_points - selected_idx) ** 2)
    plt.hist(dist, bins=10)
    # todo: plot dist vs time
    plt.figure()
    plt.plot(dist_temporal, dist)
    # TODO: embed with umap

    anchor_idx = time_points[selected_idx]
    # skip the first neighbor since the latent vector itself is always closest to itself
    nearest_idx = time_points[neighbor_idx[selected_idx]]
    # TODO: find for each label the real world example that is closest to the cluster center
    video_path = os.path.join(
        PROJECT_PATH,
        "results",
        latest_model,
        landmark_file,
        "visualization",
        "pose_sniplets_" + str(anchor_idx),
    )
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    video_file = os.path.join(video_path, "anchor.mp4")
    # last index in latent vector is the last possible time frame -> no check on last idx needed
    time_ids = np.arange(anchor_idx, anchor_idx + config["time_window"])
    create_pose_snipplet(landmark_data, landmark_names, time_ids, video_file)
    for i_neighbor, n_idx in enumerate(nearest_idx):
        video_file = os.path.join(
            video_path, f"neighbor_{i_neighbor}_" + str(n_idx) + ".mp4"
        )
        time_ids = np.arange(n_idx, n_idx + config["time_window"])

        create_pose_snipplet(landmark_data, landmark_names, time_ids, video_file)

    # TODO: create two triplet sets:
    # 1) sample any pair of same label; comp with video sniplets of cluster centers
    # 2) select any sample and show its N nearest neighbors

    # TODO: add some temporal visualization (how far are the seq. appart)

    # TODO: assumption the transition between successive frames should be smooth in the latent space
    # calc Euclidean dist between successive embeddings(since x-\hat{x} was learned)
    # visualize the distances over time
