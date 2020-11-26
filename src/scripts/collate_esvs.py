import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np

from array_ops import select

parser = argparse.ArgumentParser(
    description="Combine multiple ESV results into a single file for use with the ESV dashboard",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("esv_result_pkls", nargs="+", help="Path to ESV result pickle")
parser.add_argument(
    "collated_esv_results_pkl", help="Path to save collated ESV results to"
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Name of dataset to put into collated file. Used in ESV dashboard",
)
parser.add_argument(
    "--model",
    type=str,
    help="Name of model to put into collated file. Used in ESV dashboard",
)
parser.add_argument(
    "--uids-csv",
    type=Path,
    help="Path to CSV file containing a 'uid' column. This is used to subset the "
    "results and only aggregate a portion of the results.",
)


def main(args):
    result_paths = args.esv_result_pkls
    results: List[Dict[str, Any]] = [pd.read_pickle(path) for path in result_paths]
    if args.uids_csv is not None:
        uids: np.ndarray = pd.read_csv(args.uids_csv, converters={"uid": str})[
            "uid"
        ].values
        check_uids_are_present(result_paths, results, uids)
    else:
        uids = compute_common_uids(results)

    collated_results = collate_results(results, uids)
    collated_results["attrs"] = {}
    if args.dataset is not None:
        collated_results["attrs"]["dataset"] = args.dataset
    if args.model is not None:
        collated_results["attrs"]["model"] = args.model

    with open(args.collated_esv_results_pkl, "wb") as f:
        pickle.dump(collated_results, f)


def compute_common_uids(results: List[Dict[str, Any]]) -> np.ndarray:
    result_uids = [set(r["uids"]) for r in results]
    common_uids_set = set.union(*result_uids)
    # preserve the order of the examples based on the first result
    common_uids = np.array(
        [uid for uid in results[0]["uids"] if uid in common_uids_set]
    )
    return common_uids


def collate_results(results: List[Dict[str, Any]], uids: np.ndarray) -> Dict[str, Any]:
    first_result = results[0]
    collated_results = {
        "uids": uids,
        "labels": select(first_result["labels"], first_result["uids"], uids),
        "sequence_lengths": select(
            first_result["sequence_lengths"], first_result["uids"], uids
        ),
        "priors": first_result["priors"],
    }

    def subsample_results(key: str) -> Union[np.ndarray, List[np.ndarray]]:
        arrays = [select(result[key], result["uids"], uids) for result in results]
        try:
            return np.stack(arrays)
        except ValueError:
            # Can't stack arrays since they aren't all the same shape
            return arrays

    for k in ["scores", "sequence_idxs", "shapley_values"]:
        collated_results[k] = subsample_results(k)
    return collated_results


def check_uids_are_present(
    result_paths: List[Path], results: List[Dict[str, Any]], uids: np.ndarray
):
    quit = False
    for result_dict, result_path in zip(results, result_paths):
        uids_set = set(uids)
        result_uids_set = set(result_dict["uids"])
        missing_uids = uids_set - result_uids_set
        if len(missing_uids) > 0:
            print(f"Missing {len(missing_uids)} from {result_path}")
            print(missing_uids)
            print()
            quit = True
    if quit:
        sys.exit(1)


if __name__ == "__main__":
    main(parser.parse_args())
