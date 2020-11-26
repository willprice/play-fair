import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

import h5py
import torch
from torchvideo.samplers import FullVideoSampler
from torch.utils.data import DataLoader, Dataset as TorchDataset

from config.application import RGBConfig
from config.jsonnet import load_jsonnet
from features import FeatureExtractor, HdfFeatureWriter, ModelFeatureExtractorWrapper
from models.aggregated_backbone_model import AggregatedBackboneModel

parser = argparse.ArgumentParser(
    description="Extract per-frame features from a given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("cfg", type=Path, help="Path to configuration file")
parser.add_argument(
    "features_hdf", type=Path, help="Path to HDF5 file to save " "features to"
)
parser.add_argument("--n-workers", type=int, default=cpu_count())
parser.add_argument(
    "--split", choices=["train", "validation"], action="append", dest="splits"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="Maximum number of frames to run through backbone 2D CNN at a time. Tweak "
    "this if you're running into CUDA OOM errors.",
)


def main(args):
    logging.basicConfig(level=logging.DEBUG)
    if args.splits is None:
        print("At least one dataset split must be specified using --split")
        import sys
        sys.exit(1)
    cfg = RGBConfig(**load_jsonnet(args.cfg))
    model: AggregatedBackboneModel = cfg.get_model()
    backbone = model.backbone.eval()

    device = torch.device("cuda")
    backbone = torch.nn.DataParallel(backbone).to(device)

    datasets = get_datasets(cfg, args.splits)
    dataloaders = {
        name: DataLoader(
            dataset,
            # sadly since we are dealing with tensors of variable size we have to set
            # batch size to 1 unless we wish to deal with packing and unpacking which
            # is a massive pain.
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=args.n_workers,
        )
        for name, dataset in datasets.items()
    }

    feature_extractor = FeatureExtractor(
        backbone_2d=backbone, device=device, frame_batch_size=args.batch_size
    )
    total_instances = extract_features_to_hdf(
        dataloaders, feature_extractor, args.features_hdf, cfg.model.backbone_dim
    )
    print(f"Extracted {total_instances} features.")


def get_datasets(cfg: RGBConfig, splits: List[str]) -> Dict[str, TorchDataset]:
    dataset_builder = cfg.get_dataset()
    test_transform = cfg.get_transforms()[1]
    sampler = FullVideoSampler()

    datasets = dict()

    if "train" in splits:
        datasets["train"] = dataset_builder.train_dataset(
            transform=test_transform, sampler=sampler
        )
    if "validation" in splits:
        datasets["validation"] = dataset_builder.validation_dataset(
            transform=test_transform, sampler=sampler
        )

    return datasets


def extract_features_to_hdf(
    dataloaders: Dict[str, DataLoader],
    feature_extractor: FeatureExtractor,
    features_path: Path,
    feature_dim: int,
):
    total_instances = 0
    with h5py.File(features_path, mode="w", swmr=True, libver="latest") as root_group:
        for dataset_name, dataloader in dataloaders.items():
            n_examples = len(dataloader.dataset)
            feature_writer = HdfFeatureWriter(
                root_group.create_group(dataset_name),
                n_examples,
                feature_dim,
            )
            total_instances += feature_extractor.extract(dataloader, feature_writer)
    return total_instances


if __name__ == "__main__":
    main(parser.parse_args())
