"""Utilities to analyze the latent space."""
import cv2 as cv
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb
import pandas as pd
import numpy as np
import os
from vame.util.align_egocentrical import interpol
import tqdm
from pathlib import Path
from vame.analysis.kinutils import KinVideo, create_grid_video
from scipy.spatial.distance import cdist


def create_pose_snipplet(
    aligned_landmark_data: np.array,
    landmark_names: list,
    time_idx: np.array,
    video_path: str,
    crop_size: tuple = (400, 400),
    fps: int = 120,
):
    """Create a snipplet video of aligned landmarks.

    Args:
        aligned_landmark_data (np.array): aligned landmark data
        landmark_names (list): names of the landmarks
        time_idx (np.array): time ids based on which the video snipplet will be created
        video_path (str): path where to write the resulting video
        crop_size (tuple, optional): size of the video snipplet. Defaults to (400, 400).
        fps (int, optional): frames per second of the video snipplet. Defaults to 120.

    """
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
    save_aligned_video_path: str,
    crop_size=(300, 300),
    confidence=0.1,
):
    """Create aligned mouse video based on the landmark information.

    Args:
        video_file (str): path to the orginal video
        landmark_file (str): path to the landmarks based on which to align the data
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

    frame_count = len(data)
    align_mouse_video(
        video_file,
        save_aligned_video_path,
        crop_size,
        pose_list,
        confidence,
        frame_count,
    )

    return


def align_mouse_video(
    video_file, save_aligned_video_path, crop_size, pose_list, confidence, frame_count,
):
    # returns: list of cropped images (if video is used) and list of cropped DLC points
    #
    # parameters:
    # project_dir: project directory
    # video_file: path to the video file to process
    # save_aligned_video_path: path where the aligned video file will be stored
    # crop_size: tuple of x and y crop size
    # pose_list: list of arrays containg corresponding x and y DLC values
    # confidence: threshold based on which landmarks will be replaced by an interpolation of temporally close landmarks
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
        cv.VideoWriter_fourcc(*get_fourcc(capture)),
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


def create_visual_comparison(
    anchor_idx: int,
    latent_vectors: np.array,
    min_frame_distance: int,
    video_file: str,
    clip_length: float,
    upper_dist_percentile: int = 80,
    time_idx: list = None,
    return_sampled_idx: bool = False,
):
    """Create two videos with little video clips arranged in a matrix, where the first video shows
    the video clip corresponding to the selected anchor embedding together with the video 
    clips corresponding to its nearest neighbors in the latent space. The second video shows the anchor
    video clip and video clips corresponding to the embeddings that belong to the upper_dist_percentile percentile
    concerning their distance wrt. the anchor embedding.

    Args:
        anchor_idx (int): row index to select as anchor embedding
        latent_vectors (np.array): a matrix of shape (N_timepoints, M_embedding) where each row represents an embedding vector
        min_frame_distance (int): min distance in frames between anchor and close neighbors as well as between close neighbor since the latent
                                    vectors correspond with highly overlapping time series in the temporal domain
        video_file (str): path to the video file from which the latent vectors where predicted
        clip_length (float): lenght of the video clips in seconds
        upper_dist_percentile (int, optional): Sampling of the distant embeddings from the upper distance percentile. Defaults to 80.
        time_idx (list, optional): If a list of time_idx is provided they will be used as time index
        return_sampled_idx (bool, optional): if true return the time idx of the selected samples
    """
    n_samples = 8

    if time_idx is None:
        time_points = np.arange(0, latent_vectors.shape[0])
    else:
        time_points = time_idx

    time_anchor_idx = time_points[anchor_idx]
    selected_latent_vector = latent_vectors[anchor_idx, :]

    all_distances = np.sqrt(
        np.sum((latent_vectors - selected_latent_vector.reshape(1, -1)) ** 2, axis=1)
    )

    is_distant_to_anchor = (
        np.abs(time_points - time_points[anchor_idx]) > min_frame_distance
    )
    time_points = time_points[is_distant_to_anchor]
    latent_vectors = latent_vectors[is_distant_to_anchor]
    # distances between each latent vector and the selected one excluding the distances of latent vectors corresponding to temporally close frames
    dist = all_distances[is_distant_to_anchor]

    # select n neighbors, and enshure the neighbors are separated by a min time span
    selected_neighbor_idx = []
    while len(selected_neighbor_idx) < n_samples and len(dist) > 0:
        n_idx = np.argmin(dist)
        selected_neighbor_idx.append(time_points[n_idx])
        # remove all distances close to the selected anchor
        is_far_away = np.abs(time_points - time_points[n_idx]) > min_frame_distance
        dist = dist[is_far_away]
        time_points = time_points[is_far_away]

    samples_close = [time_anchor_idx, *selected_neighbor_idx]

    # generate "matrix" of video clips
    camera_pos, video_name = Path(video_file).parts[-2:]
    video = KinVideo(video_file, view=camera_pos)
    video.probevid()
    video_clip_data = [
        (video_file, t_id / video.getfps(), (0, 0, video.width, video.height))
        for t_id in samples_close
    ]
    grid_video_name_close = create_grid_video(video_clip_data, clip_length, speed=0.5)

    # sample latent vectors which are far away from the anchor embedding in the latent space
    dist_thr = np.percentile(dist, upper_dist_percentile)
    time_idx_other = time_points[dist > dist_thr]
    selected_distant_idx = np.random.choice(time_idx_other, 8, replace=False)

    samples_distant = [time_anchor_idx, *selected_distant_idx]
    # generate "matrix" of video clips
    video_clip_data_distant = [
        (video_file, t_id / video.getfps(), (0, 0, video.width, video.height))
        for t_id in samples_distant
    ]
    print(video_clip_data)
    grid_video_name_distant = create_grid_video(
        video_clip_data_distant, clip_length, speed=0.5
    )
    if return_sampled_idx:
        return (
            grid_video_name_close,
            grid_video_name_distant,
            samples_close,
            samples_distant,
        )
    return grid_video_name_close, grid_video_name_distant


def thin_dataset_iteratively(
    latent_vectors: np.array,
    remaining_fraction: float,
    neighbor_percentile: float,
    min_frame_distance: int,
):
    """Reduce the strong temporal dependence of embeddings close in the latent space
    by sampling repeatety an embedding and removing its temproally nearest neighbors
    if they are also closest to it in the latent space

    Args:
        latent_vectors (np.array): a matrix of shape (N_timepoints, M_embedding) where each row represents an embedding vector
        remaining_percentage (float): breaking condition to stop the thinning of data
        neighbor_percentile (float): 0...100 - percentage of latent vectors that will be based on their distance to
                                    the sampled anchor considered as neighbors. Based on this selected neighbors all embeddings that are temporally
                                    too close to the sampled embedding will be removed
        min_frame_distance (int): minimum temporal distance between a neighboring embedding to the selected anchor to be not removed from the
        dataset
    """

    time_points = np.arange(0, latent_vectors.shape[0])
    remaining_vectors = np.ones(latent_vectors.shape[0]).astype(bool)
    untested_vectors = np.ones(latent_vectors.shape[0]).astype(bool)
    sampled_anchors = []
    min_data_size = latent_vectors.shape[0] * remaining_fraction
    counter = 0
    while np.sum(untested_vectors) > 0 and np.sum(remaining_vectors) > min_data_size:
        sampled_idx = np.random.choice(time_points[untested_vectors])
        untested_vectors[sampled_idx] = False
        sampled_anchors.append(sampled_idx)
        dist = np.sqrt(
            np.sum(
                (
                    latent_vectors[remaining_vectors]
                    - latent_vectors[sampled_idx].reshape(1, -1)
                )
                ** 2,
                axis=1,
            )
        )

        dist_thr = np.percentile(dist, neighbor_percentile)
        neighbor_idx = time_points[remaining_vectors][dist < dist_thr]
        neighbor_idx = neighbor_idx[neighbor_idx != sampled_idx]

        temp_diff = np.abs(neighbor_idx - sampled_idx)
        temp_close_neighbors = neighbor_idx[temp_diff < min_frame_distance]

        # keep the vectors that where previously sampled as anchors
        temp_close_neighbors = temp_close_neighbors[
            ~np.isin(temp_close_neighbors, np.array(sampled_idx))
        ]
        # reduce time points
        remaining_vectors[temp_close_neighbors] = False
        untested_vectors[temp_close_neighbors] = False

        counter += 1
        if counter > 500:
            print(
                f"Untested: {sum(untested_vectors)}, Remaining: {sum(remaining_vectors)}"
            )
            counter = 0

    return latent_vectors[remaining_vectors], time_points[remaining_vectors]


def find_percentile_threshold(
    latent_vectors: np.array,
    time_window: int,
    time_idx: list = None,
    test_fraction: float = 0.01,
):
    """Compute the percentile in which temporally close neighbors dominate the
     neighborhood in the latent space (>=0.5 of the closest neighbors in the latent space are
     from temporally very close (and usually strongly overlapping) input time series.
     This neighbor percentile can then be used to dilute samples.

    Args:
        latent_vectors (np.array): a matrix of shape (N_timepoints, M_embedding) where each row represents an embedding vector
        time_idx (list, optional): If provided time idx of the corresponding temporal frames, otherwise it is assumed that neighboring latent vectors in the array correspond with
                                    with time series that are shifted by one step. Defaults to None.
        time_window (int): min distance between the time series corresponding to the latent vectors to be not considered as temporally close
        test_fraction (float, optional): fraction of latent vectors to sample from the dataset. Defaults to 0.01
    """
    # todo: sample n% of vectors - calc how many frames
    if time_idx is None:
        time_idx = np.arange(0, len(latent_vectors))
    else:
        assert len(latent_vectors) == len(
            time_idx
        ), "Latent vectors and time idx have different lenghts"
    sampled_ids = np.random.choice(
        len(latent_vectors), int(len(latent_vectors) * test_fraction), replace=False
    )
    time_sampled_idx = time_idx[sampled_ids]
    # output shape: M_samples, N_all_vectors
    dists = cdist(latent_vectors[sampled_ids], latent_vectors)

    # sort by distance
    idx_sorted = np.argsort(dists, axis=1)
    time_idx_sorted = time_idx[idx_sorted]

    is_temp_neighbor = (
        np.abs(time_idx_sorted - time_sampled_idx.reshape(-1, 1)) < time_window
    )

    cum_neighbor_count = np.cumsum(is_temp_neighbor, axis=1)
    cum_position = np.arange(1, len(latent_vectors) + 1)

    fraction_neighbors = cum_neighbor_count / cum_position.reshape(1, -1)

    # find position in the sorted data where temp neighbors do not dominate the neighborhood in latent space
    # anymore: e.g. less than 0.5 of the neighborhood are based on temp. close neighbors

    # issue if the first neigbor is from another time point this would be idx:0
    # therefore select latest time point where this is true
    idx_neighbors = np.stack(
        np.where(fraction_neighbors > 0.5)
    )  # this provides the first idx per row
    index_neighbors = np.median(
        [
            max(idx_neighbors[1, idx_neighbors[0, :] == i_sample])
            for i_sample in range(len(sampled_ids))
        ]
    )
    neighbor_percentile = (index_neighbors + 1) / len(latent_vectors) * 100
    return neighbor_percentile


def estimate_fuzzifier(N_data_points: int, M_feat_dim: int):
    """Estimate the fuzzifier for fuzzy c-means using the empirical formula of 
    Schwämmle and Jensen. Bioinformatics (2010)
    Paper at: https://academic.oup.com/bioinformatics/article/26/22/2841/227572
    Args:
        N_data_points (int): number of datapoints
        M_feat_dim (int): feature dimension
    """

    return (
        1
        + (1418 / N_data_points + 22.05) * M_feat_dim ** -2
        + (12.33 / N_data_points + 0.243)
        * M_feat_dim ** (-0.0406 * np.log(N_data_points) - 0.1134)
    )


def fukuyama_sugeno_index(
    data: np.array, memberships: np.array, fcm_centroids: np.array, fuzziness: float
):
    """Calculate Fukuyama Sugeno index (Fukuyama and Sugeno, 1989) using the equation in the paper
    of Schwämmle and Jensen. Bioinformatics (2010): https://academic.oup.com/bioinformatics/article/26/22/2841/227572

    Args:
        data (np.array): array of shape (N_samples, M_features)
        memberships (np.array): array of shape (N_samples, K_clusters), where each entry is a membership score to a cluster with value 0...1
        fcm_centroids (np.array): array of shape (K_clusters, M_features) predicted centroids of the fuzzy c means algorithm after fitting to the data
        fuzziness (float): fuzziess score
    """

    data_mean = np.mean(data, axis=0)

    # shape (1, K_clusters)
    sq_diff_avg_centroids = np.sum(
        np.abs(data_mean.reshape(1, -1) - fcm_centroids) ** 2, axis=1
    )

    # shape (N_features, K_clusters)??
    sq_diff_sample_centroid = cdist(data, fcm_centroids) ** 2

    # shape (N_features, K_clusters)??
    diff = sq_diff_sample_centroid - sq_diff_avg_centroids  #

    u = memberships ** fuzziness  # shape N, K

    fs_index = np.sum(np.multiply(diff, u))

    return fs_index


# get video codec code
# from https://stackoverflow.com/questions/61659346/how-to-get-4-character-codec-code-for-videocapture-object-in-opencv
def get_fourcc(cap: cv.VideoCapture):
    fourcc = int(cap.get(cv.CAP_PROP_FOURCC))
    fourcc = bytes(
        [v & 255 for v in (fourcc, fourcc >> 8, fourcc >> 16, fourcc >> 24)]
    ).decode()
    return fourcc
