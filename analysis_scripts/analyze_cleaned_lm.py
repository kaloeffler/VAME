"""Visualize the aligned landmarks (already include some interpolation for landmarks
with confidence scores below the threshold)
and the landmarks cleaned for training
 (input: the aligned landmarks, 
 processing: substract mean and sd; iqr -> interpolate at outlier positions
)"""
import os
import numpy as np
from scipy.stats import iqr
from pathlib import Path
from vame.util.auxiliary import read_config
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from vame.model.create_training import interpol
import tqdm
import cv2 as cv
from matplotlib.colors import to_rgb

project_path = "/home/katharina/vame_approach/themis_tail_belly_align/"
REMOVE_CHANNELS = False


data_path = os.path.join(project_path, "data")
landmark_dirs = [
    os.path.join(data_path, element)
    for element in os.listdir(data_path)
    if element != "train"
]
landmark_seq_files = [
    os.path.join(lm_dir, file)
    for lm_dir in landmark_dirs
    for file in os.listdir(lm_dir)
    if file.endswith("-seq.npy")
]

config_file = os.path.join(project_path, "config.yaml")
config = read_config(config_file)


# follow the same preprocessing steps as for training data creation
for file in landmark_seq_files:
    print(f"Dataset: {Path(file).parent}")
    data = np.load(file)

    # fixme: perhaps mean at least over x,y preferred? / or mean per time series?
    x_mean = np.mean(data, axis=None)
    x_std = np.std(data, axis=None)
    print(f"mean: {x_mean}; std: {x_std}")
    x_normalized = (data.T - x_mean) / x_std
    iqr_val = iqr(x_normalized)

    values_outside = np.any(
        x_normalized > config["iqr_factor"] * iqr_val, axis=1
    ) | np.any(x_normalized < -config["iqr_factor"] * iqr_val, axis=1)

    iqr_factor = config["iqr_factor"]
    print(f"Num values outside +-{iqr_factor}*{iqr_val}: {sum(values_outside)}")
    if sum(values_outside) > 0:
        x_normalized[values_outside, :] = interpol(x_normalized[values_outside, :])

    if REMOVE_CHANNELS:
        detect_anchors = np.std(data, axis=1)
        sort_anchors = np.sort(detect_anchors)
        if sort_anchors[0] == sort_anchors[1]:
            anchors = np.where(detect_anchors == sort_anchors[0])[0]
            anchor_1_temp = anchors[0]
            anchor_2_temp = anchors[1]

        else:
            anchor_1_temp = int(np.where(detect_anchors == sort_anchors[0])[0])
            anchor_2_temp = int(np.where(detect_anchors == sort_anchors[1])[0])

        if anchor_1_temp > anchor_2_temp:
            anchor_1 = anchor_1_temp
            anchor_2 = anchor_2_temp

        else:
            anchor_1 = anchor_2_temp
            anchor_2 = anchor_1_temp

        X = np.delete(X, anchor_1, 1)
        X = np.delete(X, anchor_2, 1)

        X = X.T

    x_smoothed = scipy.signal.savgol_filter(
        x_normalized.T, config["savgol_length"], config["savgol_order"]
    ).T

    # extract the landmark names
    column_names = pd.read_csv(
        os.path.join(project_path, "landmarks", Path(file).parent.name + ".csv"),
        header=[0, 1],
    ).columns
    landmark_names = [col_name[0] for col_name in column_names if col_name[-1] == "x"]

    # reshape landmarks
    x_normalized = x_normalized.T
    x_smoothed = x_smoothed.T
    x_normalized = np.transpose(
        x_normalized.reshape(-1, 2, x_normalized.shape[-1]), axes=(2, 1, 0)
    )
    x_smoothed = np.transpose(
        x_smoothed.reshape(-1, 2, x_smoothed.shape[-1]), axes=(2, 1, 0)
    )

    # create dummy plots
    # 1- landmarks before smoothing, landmarks after smoothing, overlay of the landmarks

    imgs = []
    fig, (ax_norm, ax_smooth, ax_overlay) = plt.subplots(1, 3)
    cmap = cm.get_cmap("tab20").colors

    dummy_frame = np.ones((400, 1100))
    writer = cv.VideoWriter(
        os.path.join(
            project_path, "results", Path(file).parent.name, "landmarks_cleaned.mp4"
        ),
        cv.VideoWriter_fourcc(*"DIVX"),
        60,
        dummy_frame.shape[::-1],  # xy size need to be swapped here!
    )

    # add black background
    crop_size = (300, 300)
    spacing = 50
    pos_normalized = spacing
    pos_smooth = crop_size[1] + pos_normalized + spacing
    pos_overlay = pos_smooth + crop_size[1] + spacing
    dummy_frame[
        spacing : spacing + crop_size[0], pos_normalized : pos_normalized + crop_size[1]
    ] = 0
    dummy_frame[
        spacing : spacing + crop_size[0], pos_smooth : pos_smooth + crop_size[1]
    ] = 0
    dummy_frame[
        spacing : spacing + crop_size[0], pos_overlay : pos_overlay + crop_size[1]
    ] = 0

    offset_smooth = np.array([pos_smooth, spacing])
    offset_norm = np.array([pos_normalized, spacing])
    offset_overlay = np.array([pos_overlay, spacing])
    offset_titles = np.array([0, -10])
    for i_frame, (x_norm, x_smooth) in tqdm.tqdm(
        enumerate(zip(x_normalized, x_smoothed)),
        disable=not True,
        desc="Visualize landmarkfiles",
    ):

        frame = cv.cvtColor((dummy_frame * 255).astype("uint8"), cv.COLOR_GRAY2BGR)
        cv.putText(
            frame,
            "Aligned Landmarks",
            tuple((offset_norm + offset_titles).astype(int)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        cv.putText(
            frame,
            "Landmarks After Smoothing",
            tuple((offset_smooth + offset_titles).astype(int)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        cv.putText(
            frame,
            "Overlay Landmarks Aligned / After Smoothing",
            tuple((offset_overlay + offset_titles).astype(int)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        for i_lm, (lm_norm, lm_smooth) in enumerate(zip(x_norm.T, x_smooth.T)):
            # add for visualization mean and std again
            lm_norm = (lm_norm * x_std) + x_mean
            lm_smooth = (lm_smooth * x_std) + x_mean
            # convert from rgb -> bgr
            color = tuple(
                [int(c) for c in np.array(to_rgb(cmap[i_lm % len(cmap)])[::-1]) * 255]
            )
            cv.circle(frame, tuple((lm_norm + offset_norm).astype(int)), 4, color, -1)
            cv.rectangle(
                frame,
                tuple((lm_smooth + offset_smooth).astype(int)),
                tuple((lm_smooth + offset_smooth + 4).astype(int)),
                color,
                -1,
            )

            cv.circle(
                frame, tuple((lm_norm + offset_overlay).astype(int)), 4, color, -1
            )
            cv.rectangle(
                frame,
                tuple((lm_smooth + offset_overlay).astype(int)),
                tuple((lm_smooth + offset_overlay + 4).astype(int)),
                color,
                -1,
            )

            cv.putText(
                frame,
                landmark_names[i_lm],
                tuple((lm_norm + offset_norm - 2).astype(int)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        writer.write(frame)
    writer.release()
