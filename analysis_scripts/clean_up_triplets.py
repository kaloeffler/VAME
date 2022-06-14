"""Evaluate the triplets from Shuki available at- https://github.com/ysterin/deep_cluster/tree/master/deep_cluster/triplets/data

According to Shuki: selected_triplets.csv contains the most data
the files:
selected_triplets_strong.csv
selected_triplets_robust.csv
both contain less data but clearer classifications
.
selected_tiplets_fixed.csv should have a repair for reading the data, claims there was a redundant comma or some other small mistake


"""
import pandas as pd
import os


def clean_raw_triplets_file(triplets_file: str):
    """Clean up the triplets file by removing additional header lines inbetween, removing the od index column, removing empty rows.

    Args:
        triplets_file (str): path to a triplets csv file
    """
    df = pd.read_csv(triplets_file)
    # remove replicates of the header line
    df = df[df["video_file"] != "video_file"]
    # drop original index column
    df = df.T[1:].T
    # drop nan rows
    df = df[~df["video_file"].isna()]
    return df
