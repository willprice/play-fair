import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from gulpio import GulpDirectory
from tqdm import tqdm
import PIL.Image

parser = argparse.ArgumentParser(
    description="Dump frames from gulp directory",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("gulp_dir", type=Path)
parser.add_argument("frames_root", type=Path)
parser.add_argument(
    "--quality",
    default=90,
    type=int,
    help="JPEG quality range (0--100), higher is better.",
)
parser.add_argument(
    "--uids-csv",
    type=Path,
    help="Path to CSV containing 'uid' column used to subset data",
)


def main(args):
    gulp_dir = GulpDirectory(args.gulp_dir)
    frames_root = args.frames_root
    if args.uids_csv is not None:
        uids: np.ndarray = pd.read_csv(args.uids_csv, converters={"uid": str})[
            "uid"
        ].values
    else:
        uids = np.array(list(gulp_dir.merged_meta_dict.keys()))

    for uid in tqdm(uids, dynamic_ncols=True, unit="video"):
        frames = gulp_dir[uid][0]
        frames_dir: Path = frames_root / uid
        frames_dir.mkdir(exist_ok=True, parents=True)
        for frame_idx, frame in enumerate(frames):
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            img = PIL.Image.fromarray(frame)
            img.save(frame_path, quality=args.quality)


if __name__ == "__main__":
    main(parser.parse_args())
