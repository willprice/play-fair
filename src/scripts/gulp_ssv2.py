import argparse
import gzip
import json
import os
import random
import re
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict

import pandas as pd
from gulpio import GulpIngestor
from gulpio.adapters import AbstractDatasetAdapter, Custom20BNAdapterMixin
from gulpio.utils import (
    burst_video_into_frames,
    remove_entries_with_duplicate_ids,
    resize_images,
    temp_dir_for_bursting,
)

parser = argparse.ArgumentParser(
    description="Gulp Something-Something v2 videos",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("labels_json", type=Path)
parser.add_argument(
    "--label2id-csv",
    type=Path,
    default=Path(__file__).parent.parent
    / "datasets"
    / "metadata"
    / "something_something_v2"
    / "classes.csv",
)
parser.add_argument(
    "video_dir", type=Path, help="Path to directory containing .webm SSv2 videos"
)
parser.add_argument("gulp_dir", type=Path, help="Path to write gulped videos to")
parser.add_argument(
    "--videos-per-chunk",
    type=int,
    default=1000,
    help="How many videos to store per chunk",
)
parser.add_argument(
    "--num-workers",
    "-j",
    default=cpu_count(),
    help="Number of workers to process " "the videos.",
)
parser.add_argument(
    "--frame-size",
    help="Size of smaller edge of resized frames. -1 means don't resize",
    default=-1,
)
parser.add_argument(
    "--frame-rate",
    default=8,
    type=int,
    help="Frame rate for extracted frames from video",
)
parser.add_argument(
    "--shuffle", action="store_true", help="Shuffle the dataset before ingestion"
)
parser.add_argument(
    "--extend", action="store_true", help="Extend the gulp directory with more files"
)
parser.add_argument(
    "--label-name",
    default="template",
    help="Key of the label in the JSON metadata file",
)
parser.add_argument(
    "--remove-duplicates",
    action="store_true",
    help="Remove duplicates in the JSON file and entries that are already in the "
    "gulp directory.",
)
parser.add_argument("--shm-dir", type=Path)


def main(args):
    label2id = pd.read_csv(args.label2id_csv, index_col="name")["id"].to_dict()
    adapter = JsonVideoAdapter(
        label2id=label2id,
        json_file=str(args.labels_json),
        folder=str(args.video_dir),
        output_folder=str(args.gulp_dir),
        shuffle=args.shuffle,
        frame_size=args.frame_size,
        frame_rate=args.frame_rate,
        shm_dir_path=str(args.shm_dir),
        label_name=args.label_name,
        remove_duplicate_ids=args.remove_duplicates,
    )
    ingestor = GulpIngestor(
        adapter, str(args.gulp_dir), args.videos_per_chunk, args.num_workers
    )
    ingestor()


class JsonVideoAdapter(AbstractDatasetAdapter, Custom20BNAdapterMixin):
    """Adapter for SSv2 specified by JSON file and Webm videos. A minor adaptation
    to Custom20BNJsonVideoAdapter from gulpio."""

    def __init__(
        self,
        json_file: str,
        folder: str,
        output_folder: str,
        label2id: Dict[str, int],
        shuffle: bool = False,
        frame_size: int = -1,
        frame_rate: float = 8,
        shm_dir_path: str = "/dev/shm",
        label_name: str = "template",
        remove_duplicate_ids: bool = False,
    ):
        self.json_file = json_file
        if json_file.endswith(".json.gz"):
            self.data = self.read_gz_json(json_file)
        elif json_file.endswith(".json"):
            self.data = self.read_json(json_file)
        else:
            raise RuntimeError("Wrong data file format (.json.gz or .json)")
        self.label_name = label_name
        self.output_folder = output_folder
        self.labels2idx = label2id
        self.folder = folder
        self.shuffle = bool(shuffle)
        self.frame_size = int(frame_size)
        self.frame_rate = int(frame_rate)
        self.shm_dir_path = shm_dir_path
        self.all_meta = self.get_meta()
        if remove_duplicate_ids:
            self.all_meta = remove_entries_with_duplicate_ids(
                self.output_folder, self.all_meta
            )
        if self.shuffle:
            random.shuffle(self.all_meta)

    def read_json(self, json_file):
        with open(json_file, "r") as f:
            content = json.load(f)
        return content

    def read_gz_json(self, gz_json_file):
        with gzip.open(gz_json_file, "rt") as fp:
            content = json.load(fp)
        return content

    def get_meta(self):
        return [
            {
                "id": entry["id"],
                "label": entry[self.label_name],
                "idx": self.labels2idx[
                    re.sub(r'[\[\]]', '', entry[self.label_name])
                ],
            }
            for entry in self.data
        ]

    def __len__(self):
        return len(self.all_meta)

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for meta in self.all_meta[slice_element]:
            video_path = os.path.join(self.folder, str(meta["id"]) + ".webm")
            with temp_dir_for_bursting(self.shm_dir_path) as temp_burst_dir:
                frame_paths = burst_video_into_frames(
                    video_path, temp_burst_dir, frame_rate=self.frame_rate
                )
                frames = list(resize_images(frame_paths, self.frame_size))
            result = {"meta": meta, "frames": frames, "id": meta["id"]}
            yield result
        else:
            self.write_label2idx_dict()


if __name__ == "__main__":
    main(parser.parse_args())
