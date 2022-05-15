"""Create additional visualizations to analyze the quality of the learned embeddings"""
import cv2 as cv
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb


def create_pose_snipplet(aligned_landmark_data, landmark_names, time_idx, video_path):
    # assumed shape aligned_landmark_data: n_samples, N_landmarks,2
    dummy_frame = np.zeros((400, 400))

    writer = cv.VideoWriter(
        video_path, cv.VideoWriter_fourcc(*"VP90"), 20, dummy_frame.shape[::-1]
    )
    cmap = cm.get_cmap("tab20").colors
    for time_id in time_idx:
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
