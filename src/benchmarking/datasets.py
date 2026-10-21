from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os


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


#real world data
data_path = 'data/'
if not os.path.exists(data_path):
    os.makedirs(data_path)

def get_covertype_data():
    #load covertype data
    covertype_path = data_path + 'covertype.csv'

    if os.path.exists(covertype_path):
        covertype = pd.read_csv(covertype_path)
        covertype_data = covertype.to_numpy()
    else:
        covertype = fetch_ucirepo(id=31)
        covertype.data.features.to_csv(covertype_path, index=False)
        covertype_data = covertype.data.features.to_numpy()

    covertype_2d = reduce_data(covertype_data, 2)
    covertype_3d = reduce_data(covertype_data, 3)

    return covertype_2d, covertype_3d


def get_us_census_data():
    #load us census data
    us_census_path = data_path + 'us_census_1990.csv'
    if os.path.exists(us_census_path):
        us_census = pd.read_csv(us_census_path)
        us_census_data = us_census.to_numpy()
    else:
        us_census = fetch_ucirepo(id=116)
        us_census.data.features.to_csv(us_census_path, index=False)
        us_census_data = us_census.data.features.to_numpy()

    us_census_2d = reduce_data(us_census_data, 2)
    us_census_3d = reduce_data(us_census_data, 3)
    return us_census_2d, us_census_3d