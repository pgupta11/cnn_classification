import torch
from datasets import load_dataset

def get_datasets():
    dataset_train = load_dataset('cifar10', split='train')
    dataset_val = load_dataset('cifar10', split='test')
    return dataset_train, dataset_val

def get_num_classes(dataset):
    return len(set(dataset['label']))