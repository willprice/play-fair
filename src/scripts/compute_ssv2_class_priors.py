import argparse
import re
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(
    description="Compute SSv2 class priors from empirical class frequency",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("train_json", type=Path, help="Path JSON file containing array of")
parser.add_argument(
    "class_priors_csv", type=Path, help="Path to CSV file to save class priors to."
)
parser.add_argument(
    "--label2id-csv",
    type=Path,
    default=Path(__file__).parent.parent
    / "datasets"
    / "metadata"
    / "something_something_v2"
    / "classes.csv",
)


def main(args):
    train_labels = pd.read_json(args.train_json)[["id", "template"]].rename(
        {"id": "uid", "template": "label"}, axis=1
    )
    # Replace '[something]' with 'something'
    train_labels["label"] = train_labels["label"].apply(
        lambda s: re.sub(r"[\[\]]", "", s)
    )
    label2id = pd.read_csv(args.label2id_csv, index_col="name")["id"]
    label2id.index.name = "uid"

    train_labels["class"] = label2id.loc[train_labels["label"]].values

    class_frequencies = train_labels["class"].value_counts().sort_index()
    class_priors = class_frequencies / class_frequencies.sum()
    class_priors.index.name = "class"
    class_priors.name = "prior"
    class_priors.to_csv(args.class_priors_csv)


if __name__ == "__main__":
    main(parser.parse_args())
