from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from scipy.io import arff

import os
import clip
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
from tqdm import tqdm

def get_20newsgroups():
    newsgroups = fetch_20newsgroups(subset='train')
    model = SentenceTransformer("all-mpnet-base-v2")

    data = model.encode(newsgroups.data)
    clustering_true = newsgroups.target

    return data, clustering_true

def get_caltech(root_path):
    """Code adapted from the CLIP repository, available at https://github.com/openai/CLIP"""
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Load the dataset
    full_dataset = ImageFolder(
        root=root_path,
        transform=preprocess
    )

    # 2. Find the index of the background class
    background_class_idx = full_dataset.class_to_idx['BACKGROUND_Google']

    # 3. Get the indices of all samples that are NOT the background class
    indices_to_keep = [i for i, (_, label) in enumerate(full_dataset.samples) if label != background_class_idx]

    # 4. Create a Subset using the filtered indices
    train = Subset(full_dataset, indices_to_keep)

    def get_features(dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    _data, _clustering_true = get_features(train)

def get_cifar10():
    """Code adapted from the CLIP repository, available at https://github.com/openai/CLIP"""
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Load the dataset
    root = os.path.expanduser("/content/tmp_data")
    train = CIFAR10(root, download=True, train=True, transform=preprocess)

    def get_features(dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    _data, _clustering_true = get_features(train)

    return _data, _clustering_true

def get_mnist():
    """Code adapted from the CLIP repository, available at https://github.com/openai/CLIP"""
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Load the dataset
    root = os.path.expanduser("/content/tmp_data")
    train = MNIST(root, download=True, train=True, transform=preprocess)

    def get_features(dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    _data, _clustering_true = get_features(train)

    return _data, _clustering_true

def get_beans():
    # fetch dataset
    dry_bean = fetch_ucirepo(id=602)

    # data (as pandas dataframes)
    _X = dry_bean.data.features
    _y = dry_bean.data.targets

    _data = _X.to_numpy()

    label_encoder = LabelEncoder()
    _clustering_true = label_encoder.fit_transform(_y['Class']).astype(int)

    return _data, _clustering_true

def get_wisconsin():
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    _X = breast_cancer_wisconsin_diagnostic.data.features
    _y = breast_cancer_wisconsin_diagnostic.data.targets

    _data = _X.to_numpy()

    label_encoder = LabelEncoder()
    _clustering_true = label_encoder.fit_transform(_y['Diagnosis']).astype(int)

    return _data, _clustering_true

def get_iris():
    # fetch dataset
    iris = fetch_ucirepo(id=53)

    _X = iris.data.features
    _y = iris.data.targets

    _data = _X.to_numpy()

    label_encoder = LabelEncoder()
    _clustering_true = label_encoder.fit_transform(_y['class']).astype(int)

    return _data, _clustering_true

def get_pathbased_and_r15(path_to_arff):
    """A local copy is needed for these datasets. A download is available at https://github.com/deric/clustering-benchmark"""
    raw_data, _ = arff.loadarff(path_to_arff)

    _data = np.stack([raw_data['x'],raw_data['y']], axis=-1)
    _clustering_true = raw_data['class'].astype(int)

    return _data, _clustering_true

def get_ecoli(path_to_arff):
    """A local copy is needed for these datasets. A download is available at https://github.com/deric/clustering-benchmark"""
    raw_data, _ = arff.loadarff(path_to_arff)

    _, indices, counts = np.unique(raw_data['class'], return_counts=True, return_inverse=True)
    valid = counts[indices] >= 10
    filtered_raw_data = raw_data[valid]


    numeric_cols = [name for name in filtered_raw_data.dtype.names if name != 'class']
    _data = np.stack([filtered_raw_data[name] for name in numeric_cols], axis=-1)
    _clustering_true = indices[valid]

    return _data, _clustering_true

def load_by_name(dataset_name, path = None):
    if dataset_name == '20newsgroups':
        return get_20newsgroups()
    elif dataset_name == 'cifar10':
        return get_cifar10()
    elif dataset_name == 'mnist':
        return get_mnist()
    elif dataset_name == 'beans':
        return get_beans()
    elif dataset_name == 'wisconsin':
        return get_wisconsin()
    elif dataset_name == 'iris':
        return get_iris()
    elif dataset_name == 'pathbased_and_r15':
        assert path is not None, "Path to ARFF file is required for pathbased_and_r15"
        return get_pathbased_and_r15(path)
    elif dataset_name == 'ecoli':
        assert path is not None, "Path to ARFF file is required for ecoli"
        return get_ecoli(path)
    elif dataset_name == 'caltech':
        assert path is not None, "Path to Caltech dataset is required"
        return get_caltech(path)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")