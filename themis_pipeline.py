import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import vame
from vame.initialize_project.themis_new import init_new_project
from vame.util.align_egocentrical_themis import egocentric_alignment

VIDEO_ROOT = "/media/Themis/Data/Video"
PKL_ROOT = "/media/Themis/Data/Models/3843S2B10Gaussians/analyses"

PROJECT_PATH = "/home/katharina/vame_approach/themis_dummy"

CREATE_NEW_PROJECT = False
TRAIN_MODEL = True
EVAL_MODEL = False
VISUALIZE_MODEL = False

# create landmark.csv files including the landmark positions and likelihood (confidence scores)
# similar to the DLC files and do some simple visualization of the confidence scores


# initialize new project
if CREATE_NEW_PROJECT:
    config = init_new_project(PROJECT_PATH, VIDEO_ROOT, PKL_ROOT)
else:
    config = os.path.join(PROJECT_PATH, "config.yaml")

if TRAIN_MODEL:
    # TODO: add some checks concerning the threshold - otherwise many datapoints will be rejected
    # cross check with example.csv how many datapoints are rejected

    egocentric_alignment(PROJECT_PATH, pose_ref_index=[0, 5])
    print()
    # 2.3 training data generation
    vame.create_trainset(config)

    # 3 VAME training
    vame.train_model(config)
if EVAL_MODEL:
    # 4 Evaluate trained model
    vame.evaluate_model(config)
if VISUALIZE_MODEL:
    # 5 segment motifs/pose; output latent_vector..npy file
    vame.pose_segmentation(config)

    # there are many options to for visualization
    # OPTIONIAL: Create motif videos to get insights about the fine grained poses
    vame.motif_videos(config, videoType=".mp4")

    # OPTIONAL: Create behavioural hierarchies via community detection
    vame.community(config, show_umap=True, cut_tree=2)

    # OPTIONAL: Create community videos to get insights about behavior on a hierarchical scale
    vame.community_videos(config)

    # OPTIONAL: Down projection of latent vectors and visualization via UMAP
    vame.visualization(config, label=None)  # options: label: None, "motif", "community"

    # OPTIONAL: Use the generative model (reconstruction decoder) to sample from
    # the learned data distribution, reconstruct random real samples or visualize
    # the cluster center for validation
    vame.generative_model(
        config, mode="centers"
    )  # options: mode: "sampling", "reconstruction", "centers", "motifs"
