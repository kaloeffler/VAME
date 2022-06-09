"""Pipeline to process the THEMIS landmark data from data preprocessing over model training,
 evaluation and inference (prediction of the latent vectors)"""
import os
import numpy as np
import vame
from vame.util.align_egocentrical_themis import egocentric_alignment
from datetime import datetime
from vame.model.inference import (
    inference,
    align_inference_data,
    preprocess_inference_data,
)

VIDEO_ROOT = "/media/Themis/Data/Video"
PKL_ROOT = "/media/Themis/Data/Models/3843S2B10Gaussians/analyses"

# train on K7:0044, H06:0089 ; eval on all other unseen seq.
PROJECT_PATH = "/home/katharina/vame_approach/tb_align_0089"
VIDEO_IDS_TRAINING = ["0089"]
# used to align the landmark files
# landmarks by position in the landmarks files
# ['nose', 'head', 'forepawR1', 'forepawR2', 'forePawL1',
# 'forePawL2', 'chest1', 'chest2', 'belly1', 'belly2',
# 'hindpawR1', 'hindpawR2', 'hindpawR3', 'hindpawL1',
# 'hindpawL2', 'hindpawL3', 'tailbase']
# belly1: 8, tailbase: 16
pose_alignment_idx = [8, 16]

CREATE_NEW_PROJECT = False
PREP_TRAINING_DATA = False
TRAIN_MODEL = False
EVAL_MODEL = False
PREDICT_LATENT_VECTORS = True

# create landmark.csv files including the landmark positions and likelihood (confidence scores)
# similar to the DLC files and do some simple visualization of the confidence scores

# initialize new project

# TODO: run lm_conf_quantiles.py in anaysis_scripts to see the distribution of the confidence scores in the landmark
# files and select a suitable threshold -> set the threshold in the themis_new.py : cfg_file["pose_confidence"] = 0.5
if CREATE_NEW_PROJECT:
    config = vame.init_new_project(
        PROJECT_PATH, VIDEO_ROOT, PKL_ROOT, select_video_ids=VIDEO_IDS_TRAINING
    )
else:
    config = os.path.join(PROJECT_PATH, "config.yaml")

if not os.path.exists(
    os.path.join(PROJECT_PATH, "data", "train", "pose_alignment_idx.npy")
):
    if not os.path.exists(os.path.join(PROJECT_PATH, "data", "train")):
        os.makedirs(os.path.join(PROJECT_PATH, "data", "train"))
    np.save(
        os.path.join(PROJECT_PATH, "data", "train", "pose_alignment_idx.npy"),
        np.array(pose_alignment_idx),
    )

if PREP_TRAINING_DATA:
    # landmarks by position in the landmarks files
    # ['nose', 'head', 'forepawR1', 'forepawR2', 'forePawL1',
    # 'forePawL2', 'chest1', 'chest2', 'belly1', 'belly2',
    # 'hindpawR1', 'hindpawR2', 'hindpawR3', 'hindpawL1',
    # 'hindpawL2', 'hindpawL3', 'tailbase']

    # nose: 0, tailbase: 16 - similar to what is used in VAME
    # idea: align tailbase and belly2 or chest 2 (both have high landmark confidence)
    # so that the upper body still shows rotation
    # belly1: 8, tailbase: 16
    # chest2: 7, tailbase: 16
    egocentric_alignment(
        PROJECT_PATH,
        pose_ref_index=pose_alignment_idx,
        use_video=False,
        check_video=False,
    )
    # 2.3 training data generation
    vame.create_trainset(config)

# 3: VAME training
if TRAIN_MODEL:
    vame.train_model(config)

# load the CONFIG FILE from the last trained model
trained_models = [
    (datetime.strptime(element, "%m-%d-%Y-%H-%M"), element)
    for element in os.listdir(os.path.join(PROJECT_PATH, "model"))
]
# sort by time step
trained_models.sort(key=lambda x: x[0])
latest_model = trained_models[-1][-1]

config_file = os.path.join(PROJECT_PATH, "model", latest_model, "config.yaml")

# 4: Evaluate trained model
if EVAL_MODEL:
    vame.evaluate_model(config_file)

# 5: inference (predicting latent vectors) for the whole time series
if PREDICT_LATENT_VECTORS:
    inference_path = os.path.join(PROJECT_PATH, "inference")
    train_data_dir = os.path.join(PROJECT_PATH, "data", "train")

    for lm_file in os.listdir(os.path.join(PROJECT_PATH, "landmarks")):
        lm_file = os.path.join(PROJECT_PATH, "landmarks", lm_file)
        lm_file_name = os.path.basename(lm_file).split(".")[0]

        aligned_data_file = os.path.join(
            inference_path, "data", lm_file_name, lm_file_name + "-PE-seq.npy",
        )
        if not os.path.exists(aligned_data_file):
            align_inference_data(
                lm_file,
                config_file,
                alignment_idx=pose_alignment_idx,
                save_dir=inference_path,
            )
            preprocess_inference_data(aligned_data_file, config_file, train_data_dir)
        else:
            print(f"File {aligned_data_file} already exists, skipped alignment step.")

    inference_data_files = [
        os.path.join(inference_path, "data", dir_name, dir_name + "-PE-seq-clean.npy")
        for dir_name in os.listdir(os.path.join(inference_path, "data"))
    ]

    res_path = os.path.join(inference_path, "results")
    #  load model and predict embeddings for the aligned and preprocessed data files
    inference(inference_data_files, config_file, train_data_dir, res_path)
    print(
        "Run the *.ipynb files in analysis_scripts/ folder to explore the latent space interactively"
    )
# 6) run the create_aligned_videos.py to create cropped videos for visualization

# 7) then run the *.ipynb files in analysis_scipts/ to explore the latent space interactively


#### additional visualization options form the original VAME repo ###############
# WARNING: some of the functions might not work given the changed project structure!

# there are many options to for visualization
# predict the latent vectors using the orig. VAME code
# vame.pose_segmentation(config)
# OPTIONIAL: Create motif videos to get insights about the fine grained poses
# vame.motif_videos(config, videoType=".mp4")

# OPTIONAL: Create behavioural hierarchies via community detection
# vame.community(config, show_umap=True, cut_tree=2)

# OPTIONAL: Create community videos to get insights about behavior on a hierarchical scale
# vame.community_videos(config)

# OPTIONAL: Down projection of latent vectors and visualization via UMAP
# for label in [None, "motif", "community"]:
#    vame.visualization(
#        config, label=None
#    )  # options: label: None, "motif", "community"

# OPTIONAL: Use the generative model (reconstruction decoder) to sample from
# the learned data distribution, reconstruct random real samples or visualize
# the cluster center for validation
# vame.generative_model(
#    config, mode="centers"
# )  # options: mode: "sampling", "reconstruction", "centers", "motifs"

# OPTIONAL: Create a video of an egocentrically aligned mouse + path through
# the community space (similar to our gif on github) to learn more about your representation
# and have something cool to show around ;)
# Note: This function is currently very slow. Once the frames are saved you can create a video
# or gif via e.g. ImageJ or other tools
# vame.gif(
#    config,
#    pose_ref_index=[0, 5],
#    subtract_background=False,
#    start=None,
#    length=5000,
#    max_lag=30,
#    label="community",
#    file_format=".mp4",
#    crop_size=(300, 300),
# )
