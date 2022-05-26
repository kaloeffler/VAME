import sys

sys.path.insert(0, "/home/katharina/vame_approach/VAME")
import os
import numpy as np
from vame.analysis.visualize import (
    create_aligned_mouse_video,
    create_pose_snipplet,
    thin_dataset_iteratively,
)
import pandas as pd
import cv2 as cv
from datetime import datetime
from vame.util.auxiliary import read_config

PROJECT_PATH = "/home/katharina/vame_approach/themis_tail_belly_align"
video_file = "/media/Themis/Data/Video/HO1/2021-01-11/Down/0057.MP4"
landmark_file = "/home/katharina/vame_approach/themis_tail_belly_align/landmarks/landmarks_0057_3843S2B10Gaussians_E149_confidece.csv"
align_landmark_idx = (8, 16)
save_aligned_video_path = (
    "/home/katharina/vame_approach/themis_tail_belly_align/results/align"
)

trained_models = [
    (datetime.strptime(element, "%m-%d-%Y-%H-%M"), element)
    for element in os.listdir(os.path.join(PROJECT_PATH, "model"))
]
# sort by time step
trained_models.sort(key=lambda x: x[0])
latest_model = trained_models[-1][-1]

config_file = os.path.join(PROJECT_PATH, "model", latest_model, "config.yaml")
config = read_config(config_file)


if not os.path.exists(save_aligned_video_path):
    os.makedirs(save_aligned_video_path)

create_aligned_mouse_video(
    video_file,
    landmark_file,
    align_landmark_idx,
    save_aligned_video_path,
    crop_size=(300, 300),
)

video_name = os.path.basename(video_file)
pose_video_file = os.path.join(PROJECT_PATH, "results", "poses", video_name)
if not os.path.exists(pose_video_file):
    crop_size = 400
    # min max normalize the data to a fixed grid shape for visualization
    landmark_name = os.path.basename(landmark_file).split(".")[0]
    # reshape to (N_samples, N_landmarks, 2)
    landmark_data_aligned = np.load(
        os.path.join(
            PROJECT_PATH, "data", landmark_name, landmark_name + "-PE-seq.npy"
        )
    ).T
    landmark_data_aligned = landmark_data_aligned.reshape(
        landmark_data_aligned.shape[0], -1, 2
    )
    landmark_data_trafo = (
        (landmark_data_aligned - landmark_data_aligned.min())
        / (landmark_data_aligned.max() - landmark_data_aligned.min())
        * (crop_size - 1)
    )
    capture = cv.VideoCapture(os.path.join(video_file))

    column_names = pd.read_csv(landmark_file, header=[0, 1]).columns
    landmark_names = [
        col_name[0] for col_name in column_names if col_name[-1] == "x"
    ]
    time_ids = np.arange(0, len(landmark_data_trafo))
    create_pose_snipplet(
        landmark_data_trafo,
        landmark_names,
        time_ids,
        pose_video_file,
        crop_size=(crop_size, crop_size),
        fps=capture.get(cv.CAP_PROP_FPS),
    )
