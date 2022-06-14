"""Script to generate cropped, aligned video files and videos of the landmark poses."""
import sys

sys.path.insert(0, "/home/katharina/vame_approach/VAME")
import os
import numpy as np
from vame.analysis.utils import (
    create_aligned_mouse_video,
    create_pose_snipplet,
)
import pandas as pd
import cv2 as cv
from datetime import datetime
from vame.util.auxiliary import read_config
from vame.initialize_project.themis_new import get_video_metadata

PROJECT_PATH = "/home/katharina/vame_approach/tb_align_0089"
VIDEO_ROOT = "/media/Themis/Data/Video"

save_aligned_video_path = os.path.join(PROJECT_PATH, "videos/aligned_videos")


# create a video showing the aligned landmarks only in addition to the cropped video
CREATE_POSE_VIDEO = False

for lm_file in os.listdir(os.path.join(PROJECT_PATH, "landmarks")):
    aligned_video_name = os.path.join(
        save_aligned_video_path, "a" + lm_file.split("_")[1] + ".MP4"
    )
    if os.path.exists(aligned_video_name):
        continue
    landmark_file = os.path.join(PROJECT_PATH, "landmarks", lm_file)

    video_id = os.path.basename(landmark_file).split("_")[1]
    video_info = get_video_metadata(VIDEO_ROOT, "")
    video_file = os.path.join(
        video_info[video_info["vid_file"] == video_id + ".MP4"]["vid_folder"].values[0],
        video_id + ".MP4",
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
        video_file, landmark_file, save_aligned_video_path, crop_size=(300, 300),
    )

    video_name = os.path.basename(video_file)
    if CREATE_POSE_VIDEO:
        pose_video_file = os.path.join(
            PROJECT_PATH, "videos", "poses", "p" + video_name
        )
        if not os.path.exists(pose_video_file):
            crop_size = 400
            # min max normalize the data to a fixed grid shape for visualization
            landmark_name = os.path.basename(landmark_file).split(".")[0]
            # reshape to (N_samples, N_landmarks, 2)
            landmark_data_aligned = np.load(
                os.path.join(
                    PROJECT_PATH,
                    "inference",
                    "data",
                    landmark_name,
                    landmark_name + "-PE-seq.npy",
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
