from datasets.base import DataDict, load_dataset
from datasets.cora import load_cora
from datasets.citeseer import load_citeseer
from datasets.pubmed import load_pubmed

__all__ = [
    "DataDict",
    "load_dataset",
    "load_cora",
    "load_citeseer",
    "load_pubmed",
]
