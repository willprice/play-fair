from typing import List, Union

import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def display_video(video: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
                  format="THWC", fps=12):
    """
    Args:
        video: Video array or tensor with values ranging between 0--255.
        format: TCHW in any order, describing the data layour of the video array
    """
    if isinstance(video, list):
        video = np.stack(video)
    if isinstance(video, torch.Tensor):
        video = video.numpy()
    video = np.einsum(f"{format} -> THWC", video.astype(np.uint8))
    _, height, width, _ = video.shape
    if height % 2 != 0:
        video = video[:, :-1, :, :]
    if width % 2 != 0:
        video = video[:, :, :-1, :]
    print(video.shape)
    frames: List[np.ndarray] = list(video)
    clip = ImageSequenceClip(frames, fps=fps)

    return clip.ipython_display()



