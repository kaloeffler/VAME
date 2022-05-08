from vame.util.prep_themis_data import get_video_metadata, pickle_dlc_to_df
import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn


VIDEO_ROOT = "/media/Themis/Data/Video"

# in this dir there is currently only one pkl file
PKL_ROOT = "/media/Themis/Data/Models/3843S2B10Gaussians/analyses"
VIS_RES_PATH = "/home/katharina/vame_approach/results/landmark_conf_distribution"

# where to save the create landmark.csv files that include landmark positions and their confidence
SAVE_DF_PATH = "/home/katharina/vame_approach/landmark_data"
if not os.path.exists(SAVE_DF_PATH):
    os.makedirs(SAVE_DF_PATH)

pkl_files = [
    os.path.join(PKL_ROOT, element)
    for element in os.listdir(PKL_ROOT)
    if element.endswith(".pkl")
]

video_df = get_video_metadata(VIDEO_ROOT, DLC_model="")
video_df["video_id"] = video_df["vid_file"].apply(lambda x: x.split(".")[0])
# map videos to pkl files
video_ids = [re.findall(r"\d+", pkl_file)[0] for pkl_file in pkl_files]
video_df = video_df[np.isin(video_df["video_id"].values, video_ids)]

# pkl to dataframe
for pkl_file in pkl_files:
    dlc_file = os.path.join(
        PKL_ROOT, "_".join(os.path.basename(pkl_file).split("_")[:-1]) + ".csv"
    )
    if not os.path.exists(dlc_file):
        print(f"No file named {dlc_file}")
        continue
    landmark_names = pd.read_csv(dlc_file, header=[0, 1]).columns
    df, df_conf = pickle_dlc_to_df(pkl_file, landmark_names, return_conf_df=True)
    # drop the likelihood part
    col_names = {col_name: col_name[0] for col_name in df_conf.columns}
    df_conf.rename(columns=col_names, inplace=True)

    print("--" * 10)
    print(f"Landmark Confidence Quantiles of {pkl_file}")
    print(df_conf.quantile([0.1, 0.5, 0.9]))
    print("--" * 10)

    plot = seaborn.violinplot(data=df_conf[::100], orient="h")
    plot.set(title="Distribution of Confidence Scores")
    plt.tight_layout()
    plot.get_figure().savefig(
        os.path.join(VIS_RES_PATH, os.path.basename(pkl_file).split(".")[0] + ".png")
    )

    # save file to disk
    df.to_csv(os.path.join(SAVE_DF_PATH))
