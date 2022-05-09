"""
Adapt align_egocentrial such that it works with the adapted file structure
(videos won't be copied but are kept as .csv file with paths to them and
instead of evaluation by videos the evaluation is done by the landmark predictions
since different models where trained that need to be compared)
"""
import os
import cv2 as cv
import numpy as np
import pandas as pd
import tqdm

from pathlib import Path
from vame.util.auxiliary import read_config
from vame.util.align_egocentrical import interpol, crop_and_flip, play_aligned_video
import re


def align_mouse(
    project_dir,
    landmark_file_name,
    crop_size,
    pose_list,
    pose_ref_index,
    confidence,
    pose_flip_ref,
    bg,
    frame_count,
    use_video=True,
):
    # returns: list of cropped images (if video is used) and list of cropped DLC points
    #
    # parameters:
    # project_dir: project directory
    # landmark_file_name: name of the landmark file to process
    # crop_size: tuple of x and y crop size
    # pose_list: list of arrays containg corresponding x and y DLC values
    # pose_ref_index: indices of 2 lists in dlc_list to align mouse along
    # pose_flip_ref: indices of 2 lists in dlc_list to flip mouse if flip was false
    # bg: background image to subtract
    # frame_count: number of frames to align
    # use_video: boolean if video should be cropped or DLC points only

    images = []
    points = []

    # get path to the corresponding video file
    video_df = pd.read_csv(os.path.join(project_dir, "video_info.csv"))
    video_id = int(re.findall(r"\d+", landmark_file_name)[0])
    video_file = os.path.join(
        *video_df[video_df["video_id"] == video_id][["vid_folder", "vid_file"]].values[
            0
        ]
    )

    for i in pose_list:
        for j in i:
            if j[2] <= confidence:
                j[0], j[1] = np.nan, np.nan

    for i in pose_list:
        i = interpol(i)

    if use_video:
        capture = cv.VideoCapture(os.path.join(video_file))

        if not capture.isOpened():
            raise Exception(
                "Unable to open video file: {0}".format(os.path.join(video_file))
            )

    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc="Align frames"):

        if use_video:
            # Read frame
            try:
                ret, frame = capture.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame - bg
                frame[frame <= 0] = 0
            except:
                print("Couldn't find a frame in capture.read(). #Frame: %d" % idx)
                continue
        else:
            frame = np.zeros((1, 1))

        # Read coordinates and add border
        pose_list_bordered = [
            (int(i[idx][0] + crop_size[0]), int(i[idx][1] + crop_size[1]))
            for i in pose_list
        ]

        img = cv.copyMakeBorder(
            frame,
            crop_size[1],
            crop_size[1],
            crop_size[0],
            crop_size[0],
            cv.BORDER_CONSTANT,
            0,
        )

        punkte = np.array(pose_list_bordered)[pose_ref_index, :].reshape(1, -1, 2)
        # calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)

        # change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)

        center, size, theta = rect

        # crop image
        out, shifted_points = crop_and_flip(
            rect, img, pose_list_bordered, pose_flip_ref
        )

        if use_video:  # for memory optimization, just save images when video is used.
            images.append(out)
        points.append(shifted_points)

    if use_video:
        capture.release()

    time_series = np.zeros((len(pose_list) * 2, frame_count))
    for i in range(frame_count):
        idx = 0
        for j in range(len(pose_list)):
            time_series[idx : idx + 2, i] = points[i][j]
            idx += 2

    return images, points, time_series


def egocentric_alignment(
    project_path,
    pose_ref_index=[0, 5],
    crop_size=(300, 300),
    use_video=False,
    check_video=False,
):
    """ Happy aligning """
    config = os.path.join(project_path, "config.yaml")
    video_df = pd.read_csv(os.path.join(project_path, "video_info.csv"))

    # config parameters
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    path_to_file = cfg["project_path"]
    landmark_file_names = cfg["video_sets"]
    confidence = cfg["pose_confidence"]
    crop_size = crop_size

    # call function and save into your VAME data folder
    for lm_file_name in landmark_file_names:
        print(
            "Aligning data %s, Pose confidence value: %.2f" % (lm_file_name, confidence)
        )
        video_id = int(re.findall(r"\d+", lm_file_name)[0])
        video_file = os.path.join(
            *video_df[video_df["video_id"] == video_id][
                ["vid_folder", "vid_file"]
            ].values[0]
        )
        egocentric_time_series, frames = alignment(
            video_file,
            project_path,
            lm_file_name,
            pose_ref_index,
            crop_size,
            confidence,
            use_video=use_video,
            check_video=check_video,
        )
        np.save(
            os.path.join(
                path_to_file, "data", lm_file_name, lm_file_name + "-PE-seq.npy"
            ),
            egocentric_time_series,
        )
    #        np.save(os.path.join(path_to_file,'data/',file,"",file+'-PE-seq.npy', egocentric_time_series))

    print("Your data is now ine right format and you can call vame.create_trainset()")


def alignment(
    video_file,
    project_dir,
    landmark_file_name,
    pose_ref_index,
    crop_size,
    confidence,
    use_video=False,
    check_video=False,
):

    # read out data
    data = pd.read_csv(
        os.path.join(project_dir, "landmarks", landmark_file_name + ".csv"), skiprows=2,
    )
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:, 1:]

    # get the coordinates for alignment from data table
    pose_list = []

    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])

    # list of reference coordinate indices for alignment
    # 0: snout, 1: forehand_left, 2: forehand_right,
    # 3: hindleft, 4: hindright, 5: tail

    pose_ref_index = pose_ref_index

    # list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index

    if use_video:
        # compute background
        bg = background(video_file)
        capture = cv.VideoCapture(video_file)
        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(video_file))

        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        capture.release()
    else:
        bg = 0
        frame_count = len(
            data
        )  # Change this to an abitrary number if you first want to test the code
    # FIXME: remove
    frame_count = 1000
    frames, n, time_series = align_mouse(
        project_dir,
        landmark_file_name,
        crop_size,
        pose_list,
        pose_ref_index,
        confidence,
        pose_flip_ref,
        bg,
        frame_count,
        use_video,
    )

    if check_video:
        save_video_path = os.path.join(project_dir, "results", landmark_file_name)
        if not os.path.exists(save_video_path):
            os.makedirs(save_video_path)
        df = pd.read_csv(
            os.path.join(project_dir, "landmarks", landmark_file_name + ".csv"),
            header=[0, 1],
        )
        landmark_names = [col_name[0] for col_name in df.columns if col_name[1] == "x"]
        play_aligned_video(frames, n, frame_count, landmark_names, save_video_path)

    return time_series, frames


def background(video_file, num_frames=1000):
    """
    Compute background image from fixed camera 
    """
    import scipy.ndimage

    capture = cv.VideoCapture(video_file)

    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(video_file))

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()

    height, width, _ = frame.shape
    frames = np.zeros((height, width, num_frames))

    for i in tqdm.tqdm(
        range(num_frames),
        disable=not True,
        desc="Compute background image for video %s" % video_file,
    ):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1, rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[..., i] = gray

    print("Finishing up!")
    medFrame = np.median(frames, 2)
    background = scipy.ndimage.median_filter(medFrame, (5, 5))

    # np.save(path_to_file+'videos/'+'background/'+filename+'-background.npy',background)

    capture.release()
    return background
