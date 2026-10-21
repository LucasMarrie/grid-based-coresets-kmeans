from functools import lru_cache
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os


DATASET_NAMES = [
    "2d blobs",
    "3d blobs",
    "Covertype 2d",
    "Covertype 3d",
    "US Census 2d",
    "US Census 3d",
]


#reduce dimensionality of data
def reduce_data(data, dimensions):
    pca = PCA(n_components=dimensions)
    return pca.fit_transform(data)


def get_blobs_2d(samples, centers, seed=42):
    np.random.seed(seed)
    blobs_2d, _ = make_blobs(n_samples=samples, centers=centers, n_features=2, random_state=seed)
    return blobs_2d

def get_blobs_3d(samples, centers, seed=42):
    np.random.seed(seed)
    blobs_3d, _ = make_blobs(n_samples=samples, centers=centers, n_features=3, random_state=seed)
    return blobs_3d


#real world data, downloaded from the UCI repository on first use and cached locally
data_path = 'data/'

def _load_uci_features(uci_id, filename):
    path = os.path.join(data_path, filename)
    if os.path.exists(path):
        return pd.read_csv(path).to_numpy()

    os.makedirs(data_path, exist_ok=True)
    features = fetch_ucirepo(id=uci_id).data.features
    features.to_csv(path, index=False)
    return features.to_numpy()


@lru_cache(maxsize=None)
def get_covertype_data():
    covertype_data = _load_uci_features(31, 'covertype.csv')
    return reduce_data(covertype_data, 2), reduce_data(covertype_data, 3)


@lru_cache(maxsize=None)
def get_us_census_data():
    us_census_data = _load_uci_features(116, 'us_census_1990.csv')
    return reduce_data(us_census_data, 2), reduce_data(us_census_data, 3)


def get_test_datasets(k_clusters, names=DATASET_NAMES, blob_samples=1000000):
    """
    Build the list of test dataset dicts ({'name', 'dataset', 'k_clusters'}) used by the experiments.
    Real-world datasets are only loaded if one of their variants is requested.
    """
    loaders = {
        "2d blobs": lambda: get_blobs_2d(blob_samples, k_clusters),
        "3d blobs": lambda: get_blobs_3d(blob_samples, k_clusters),
        "Covertype 2d": lambda: get_covertype_data()[0],
        "Covertype 3d": lambda: get_covertype_data()[1],
        "US Census 2d": lambda: get_us_census_data()[0],
        "US Census 3d": lambda: get_us_census_data()[1],
    }
    return [
        {"name": name, "dataset": loaders[name](), "k_clusters": k_clusters}
        for name in names
    ]
