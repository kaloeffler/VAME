"""Create additional visualizations to analyze the quality of the learned embeddings"""
import cv2 as cv
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb
from vame.util.align_egocentrical_themis import align_mouse
import pandas as pd
import numpy as np
import os
from vame.util.align_egocentrical import interpol, crop_and_flip, play_aligned_video
import tqdm


def create_pose_snipplet(
    aligned_landmark_data,
    landmark_names,
    time_idx,
    video_path,
    crop_size=(400, 400),
    fps=120,
):
    # assumed shape aligned_landmark_data: n_samples, N_landmarks,2
    dummy_frame = np.zeros(crop_size)
    if os.path.exists(video_path):
        raise AssertionError(f"Video file already exists! {video_path}")
    writer = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*"VP90"), fps, crop_size)
    cmap = cm.get_cmap("tab20").colors
    for time_id in tqdm.tqdm(time_idx, disable=not True, desc="Create pose frames"):
        landmarks = aligned_landmark_data[time_id]

        frame = cv.cvtColor((dummy_frame * 255).astype("uint8"), cv.COLOR_GRAY2BGR)

        for i_lm, lm in enumerate(landmarks):
            # add for visualization mean and std again

            # convert from rgb -> bgr
            color = tuple(
                [int(c) for c in np.array(to_rgb(cmap[i_lm % len(cmap)])[::-1]) * 255]
            )
            cv.circle(frame, tuple(lm.astype(int)), 5, color, -1)

            cv.putText(
                frame,
                landmark_names[i_lm],
                tuple((lm - 2).astype(int)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
            )
        writer.write(frame)
    # add a single empty frame to visualize in loops the start/end better
    frame = cv.cvtColor((dummy_frame * 0).astype("uint8"), cv.COLOR_GRAY2BGR)
    writer.write(frame)
    writer.release()


def create_aligned_mouse_video(
    video_file: str,
    landmark_file: str,
    align_landmark_idx: tuple,
    save_aligned_video_path: str,
    crop_size=(300, 300),
    confidence=0.1,
):
    """Create aligned mouse video based on the landmark information.

    Args:
        video_file (str): path to the orginal video
        landmark_file (str): path to the landmarks based on which to align the data
        align_landmark_idx (tuple): pair of landmarks based on which the video will be aligned
        save_aligned_video_path (str): path to the directory where the aligned video will be saved
        crop_size (tuple): size of the aligned video
    """
    # read out data
    data = pd.read_csv(os.path.join(landmark_file), skiprows=2,)
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:, 1:]

    # get the coordinates for alignment from data table
    pose_list = []
    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])

    # list of reference coordinate indices for alignment
    # 0: snout, 1: forehand_left, 2: forehand_right,
    # 3: hindleft, 4: hindright, 5: tail

    pose_ref_index = align_landmark_idx

    # list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index

    frame_count = len(data)
    align_mouse_video(
        video_file,
        save_aligned_video_path,
        crop_size,
        pose_list,
        pose_ref_index,
        confidence,
        pose_flip_ref,
        frame_count,
    )

    return


def align_mouse_video(
    video_file,
    save_aligned_video_path,
    crop_size,
    pose_list,
    pose_ref_index,
    confidence,
    pose_flip_ref,
    frame_count,
):
    # returns: list of cropped images (if video is used) and list of cropped DLC points
    #
    # parameters:
    # project_dir: project directory
    # video_file: path to the video file to process
    # save_aligned_video_path: path where the aligned video file will be stored
    # crop_size: tuple of x and y crop size
    # pose_list: list of arrays containg corresponding x and y DLC values
    # pose_ref_index: indices of 2 lists in dlc_list to align mouse along
    # pose_flip_ref: indices of 2 lists in dlc_list to flip mouse if flip was false
    # frame_count: number of frames to align

    # get path to the corresponding video file

    for i in pose_list:
        for j in i:
            if j[2] <= confidence:
                j[0], j[1] = np.nan, np.nan

    for i in pose_list:
        i = interpol(i)

    capture = cv.VideoCapture(os.path.join(video_file))

    if not capture.isOpened():
        raise Exception(
            "Unable to open video file: {0}".format(os.path.join(video_file))
        )
    video_name, file_ending = os.path.basename(video_file).split(".")
    video_writer = cv.VideoWriter(
        os.path.join(save_aligned_video_path, "a" + video_name + "." + file_ending),
        cv.VideoWriter_fourcc(*"VP90"),
        capture.get(cv.CAP_PROP_FPS),
        crop_size,
    )
    img_stack = []
    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc="Align frames"):
        try:
            ret, frame = capture.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        except:
            print("Couldn't find a frame in capture.read(). #Frame: %d" % idx)
            continue

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

        punkte = np.array(pose_list_bordered).reshape(1, -1, 2)
        # calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)

        # change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)

        # crop image
        out = crop_img(rect, img)

        frame = cv.cvtColor(out.astype("uint8"), cv.COLOR_GRAY2BGR)
        img_stack.append(frame)
        if len(img_stack) > 10000:
            for frame in img_stack:
                video_writer.write(frame)
            img_stack = []
    for frame in img_stack:
        video_writer.write(frame)
    capture.release()
    video_writer.release()
    return


def crop_img(rect, src):
    center, size, _ = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    out = cv.getRectSubPix(src, size, center)

    return out
