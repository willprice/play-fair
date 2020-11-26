import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import numpy as np
import torch
from torchvideo.samplers import frame_idx_to_list
from tqdm import tqdm

from attribution.online_shapley_value_attributor import OnlineShapleyAttributor
from attribution.characteristic_function_shapley_value_attributor import (
    CharacteristicFunctionShapleyAttributor,
)
from config.application import FeatureConfig, RGBConfig
from config.jsonnet import load_jsonnet
from frame_sampling import RandomSampler
from subset_samplers import ConstructiveRandomSampler, ExhaustiveSubsetSampler

parser = argparse.ArgumentParser(
    description="Compute ESVs given a trained model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("cfg", type=Path, help="Path to model jsonnet configuration")
parser.add_argument(
    "class_priors_csv",
    type=Path,
    help="Path to CSV file containing class priors to use to represent the output of "
    "f({})",
)
parser.add_argument(
    "esv_pkl",
    type=Path,
    help="Path to save model results and frame ESVs to (as a pickle)",
)
parser.add_argument(
    "--sample-n-frames",
    type=int,
    default=8,
    help="How many frames to sample from the video. -1 means computing ESVs for all "
    "frames in a video, you will have to enable approximation in almost all cases if "
    "you do this.",
)
parser.add_argument(
    "--approximate", action="store_true", help="Whether to approximate ESV or not."
)
parser.add_argument(
    "--approximate-n-iters",
    default=1,
    type=int,
    help="Number of iterations of outer loop that repeatedly approximates multiscale "
    "model and ESVs. Runtime: O(n_iters)",
)
parser.add_argument(
    "--approximate-max-samples-per-scale",
    default=128,
    type=int,
    help="Maximum number of subsequences to sample for each subsequence length. This "
    "will depend on how long the video you are computing ESVs for.",
)


def main(args):
    model_cfg_dict = load_jsonnet(args.cfg)
    if "features" not in model_cfg_dict["dataset"]["kind"].lower():
        raise ValueError(
            "Computing ESVs for RGB model takes too long, extract "
            "features first then apply compute ESVs."
        )
    cfg = FeatureConfig(**model_cfg_dict)
    class_priors = pd.read_csv(args.class_priors_csv, index_col="class")['prior'].values
    device = torch.device("cuda:0")
    n_frames = args.sample_n_frames
    frame_sampler = RandomSampler(frame_count=n_frames, snippet_length=1, test=True)
    subset_sampler = get_subset_sampler(args, device)
    dataset_config = cfg.get_dataset()
    model = cfg.get_model().eval().to(device)
    dataset = dataset_config.validation_dataset()

    def subsample_frames(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        video_length = len(video)
        if video_length < n_frames:
            raise ValueError(f"Video too short to sample {n_frames} from")
        sample_idxs = np.array(frame_idx_to_list(frame_sampler.sample(video_length)))
        return sample_idxs, video[sample_idxs]

    class_count = dataset_config.class_count()
    if hasattr(model, "single_scale_models"):
        # Use our multiscale shapley value analysis
        attributor = OnlineShapleyAttributor(
            single_scale_models=model.single_scale_models,
            priors=class_priors,
            n_classes=class_count,
            device=device,
            subset_sampler=subset_sampler,
        )
    else:
        print(
            "Assuming model supports variable length inputs. If that is not the "
            "case, then your model must be have a single_scale_models attribute."
        )
        device_priors = torch.tensor([class_priors], device=device)

        # Not this is not *quite* the charactestic function you'd get in the original
        # Shapley value formulation since we don't subtract f(X) by f(\emptyset).
        # Since in only one case (when we're considering inputs of length 1) we need to
        # do that, we bake that into the CharacteristicFunctionShapleyAttributor class
        # The contract of this function is that it should return the model output on
        # the batched sequences provided if they have at least one or more elements.
        # If there are 0 elements, then the function should return the class priors
        def characteristic_fn(batch_subseq_features):
            assert batch_subseq_features.ndim == 3
            if batch_subseq_features.shape[1] == 0:
                out = device_priors
            else:
                with torch.no_grad():
                    out = torch.softmax(model(batch_subseq_features), dim=-1)
            assert out.ndim == 2, f"expected output to be 2D but had shape {out.shape}."
            assert out.shape[0] == len(batch_subseq_features), (
                f"Batch count changed from {len(batch_subseq_features)} to "
                f"{out.shape[0]}."
            )
            return out

        attributor = CharacteristicFunctionShapleyAttributor(
            characteristic_fn=characteristic_fn,
            n_classes=class_count,
            subset_sampler=subset_sampler,
            device=device,
        )

    data = {
        "labels": [],
        "uids": [],
        "sequence_idxs": [],
        "sequence_lengths": [],
        "scores": [],
        "shapley_values": [],
    }
    for i, (video, label_dict) in enumerate(
        tqdm(dataset, dynamic_ncols=True, unit="video")
    ):
        cls = label_dict["action"]
        uid = label_dict["uid"]
        try:
            subsample_idxs, subsampled_video = subsample_frames(video)
            device_subsampled_video = torch.from_numpy(subsampled_video).to(device)
        except ValueError:
            print(
                f"{uid} is too short ({len(video)} frames) to sample {n_frames} "
                f"frames from."
            )
            continue

        esvs, scores = attributor.explain(device_subsampled_video)

        data["labels"].append(cls)
        data["uids"].append(uid)
        data["sequence_idxs"].append(subsample_idxs)
        data["sequence_lengths"].append(len(video))
        data["scores"].append(scores.cpu().numpy())
        data["shapley_values"].append(esvs.cpu().numpy())

    def collate(vs: List[Any]):
        # Convert to np array if we can, otherwise leave as list
        try:
            return np.stack(vs)
        except ValueError:
            return vs

    data_to_persist = {k: collate(vs) for k, vs in data.items()}
    data_to_persist['priors'] = class_priors
    with open(args.esv_pkl, "wb") as f:
        pickle.dump(data_to_persist, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_subset_sampler(args, device):
    if args.approximate:
        return ConstructiveRandomSampler(
            max_samples=args.approximate_max_samples_per_scale, device=device
        )
    return ExhaustiveSubsetSampler(device=device)


if __name__ == "__main__":
    main(parser.parse_args())
