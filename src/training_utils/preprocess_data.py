import datasets
from datasets import inspect_dataset, load_dataset_builder

inspect_dataset("wmt14", "path/to/scripts")
builder = load_dataset_builder(
    "path/to/scripts/wmt_utils.py",
    language_pair=("fr", "de"),
    subsets={
        datasets.Split.TRAIN: ["commoncrawl_frde"],
        datasets.Split.VALIDATION: ["euelections_dev2019"],
    },
)

# Standard version
builder.download_and_prepare()
ds = builder.as_dataset()

print(ds)