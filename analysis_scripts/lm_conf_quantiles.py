"""
Calc the quantiles of the landmark confidences, because the VAME approach requires a threshold based on
which datapoints will be rejected; the threshold is referred to as "pose_confidence" parameter in config.yaml file
values below the threshold will be set to nan and interpolated in the egocentric alignment function!
"""
import os
import pandas as pd
import numpy as np

LANDMARK_DIR = "/home/katharina/vame_approach/themis_dummy/landmarks"

landmark_files = [os.path.join(LANDMARK_DIR, file) for file in os.listdir(LANDMARK_DIR)]

demo_df = pd.read_csv(
    "/home/katharina/vame_approach/VAME/examples/video-1.csv", header=[1, 2]
)
is_conf_column = [col_name[-1] == "likelihood" for col_name in demo_df.columns]
demo_df_conf = demo_df[demo_df.columns[is_conf_column]]

print("-" * 20)
print("Confidence scores of demo file")
print(demo_df_conf.quantile(np.linspace(0.1, 0.9, 9)))
print("-" * 20)


for lm_file in landmark_files:
    df = pd.read_csv(lm_file, header=[0, 1])
    is_conf_column = [col_name[-1] == "likelihood" for col_name in df.columns]
    df_conf = df[df.columns[is_conf_column]]
    print("-" * 20)
    print(f"Confidence scores of {lm_file}")
    print(df_conf.quantile(np.linspace(0.1, 0.9, 9)))
    print("-" * 20)