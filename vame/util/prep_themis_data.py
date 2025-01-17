import os
import pandas as pd
import numpy as np
import pickle
from glob import glob


def get_video_metadata(
    video_root: str, DLC_model: str = "DLC_resnet50_dlc_testJan21shuffle1_380600.h5"
):
    """Creates a DataFrame of video metadata based on files under the video_root folder
    
    Args:
        video_root (str): path to the root folder of the rat videos
        DLC_model (str, optional): Return only videos that have a specific model for landmark prediction already trained on. Defaults to "DLC_resnet50_dlc_testJan21shuffle1_380600.h5".

    """

    vid_list = [
        os.path.split(path)
        for path in glob(f"{video_root}/**/Down/0*.MP4", recursive=True)
    ]
    dlc_list = [
        os.path.split(path)
        for path in glob(f"{video_root}/**/Down/*{DLC_model}*.h5", recursive=True)
    ]

    vid_df = pd.DataFrame(columns=["rat", "date", "vid_folder", "vid_file", "dlc_file"])
    for vid in vid_list:
        vid_folder = vid[0]
        vid_file = vid[1]
        path_pieces = os.path.normpath(vid[0]).split(os.path.sep)
        rat = path_pieces[-3]
        date = path_pieces[-2]
        found = False
        for path in dlc_list:
            if vid[1][:4] in path[1]:
                dlc_file = path[1]
                found = True
                continue
        if not found:
            dlc_file = np.nan
        vid_df = vid_df.append(
            {
                "rat": rat,
                "date": date,
                "vid_folder": vid_folder,
                "vid_file": vid_file,
                "dlc_file": dlc_file,
            },
            ignore_index=True,
        )

    # vid_df = vid_df.dropna()
    vid_df.reset_index(drop=True, inplace=True)
    return vid_df


def pickle_dlc_to_df(pkl_file: str, df_header: list, return_conf_df: bool = False):
    """Convert pkl file with landmarks and confidence measures to a dataframe similar as in DLC.

    Args:
        pkl_file (str): path to the pickle file containing landmarks and confidence scores
        df_header (list): names of the landmarks
        return_conf_df (bool, optional): if true return the confidence scores as a dataframe. Defaults to False.
    """
    with open(pkl_file, "rb") as file:
        data = pickle.load(file)
    landmark_positions = data["orig_lm"]
    landmark_confidence = data["confidence"]["conf"]
    # create nested header landmark_name: x,y, likelihood
    lm_index, landmark_names = df_header[0], df_header[1:]

    landmark_position_df = pd.DataFrame(
        landmark_positions.reshape(landmark_positions.shape[0], -1),
        columns=landmark_names,
    )
    # keep the same order of the landmarks and replace dim with confidence
    # - one confidence score per landmark (x,y pair)
    landmark_conf_names = [
        (lm[0], "likelihood") for lm in landmark_names if lm[-1] == "x"
    ]
    landmark_conf_df = pd.DataFrame(landmark_confidence, columns=landmark_conf_names)

    landmark_df = pd.concat([landmark_position_df, landmark_conf_df], axis=1)

    # rearrange df such that bodypart1: x,y, likelihood; bodypart2:x,y,likelihood
    columns_sorted = [
        (lm[0], dim) for lm in landmark_conf_names for dim in ["x", "y", "likelihood"]
    ]

    landmark_df = landmark_df[columns_sorted]
    landmark_df.index.name = lm_index
    if return_conf_df:
        return landmark_df, landmark_conf_df
    return landmark_df
