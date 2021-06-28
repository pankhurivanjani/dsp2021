# 3
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

seeds_dataset_file = open('DSP_Assignment_10/seeds_dataset.txt', 'r')
seeds_dataset = np.array([list(map(float, line.strip('\n').split())) for line in seeds_dataset_file])

n_features = 7
n_classes = len(np.unique(seeds_dataset[...,7]))
 
# 3.1
def get_mean_per_class(seeds_dataset):
    sum_per_class = np.zeros((n_classes, n_features))
    count_per_class = np.zeros((n_classes))
    for seed in seeds_dataset:
        class_idx = seed[7].astype(int)-1
        sum_per_class[class_idx] += seed[:n_features]
        count_per_class[class_idx] += 1
    return sum_per_class/count_per_class[...,None]


    
# 3.2
def get_within_class_scatter_matrix(seeds_dataset, mean_per_class):
    class_idxs, class_counts = np.unique(seeds_dataset[...,7], return_counts=True)
    fraction_classes = class_counts / class_counts.sum()
    within_class_scatter_matrix = np.zeros((n_features, n_features))
    for class_idx in class_idxs: 
        class_idx = int(class_idx - 1)
        for seed in seeds_dataset:
            if int(seed[...,7] - 1.) == class_idx:
                sigma_class = (seed[:n_features] - mean_per_class[class_idx]) * (seed[:n_features] - mean_per_class[class_idx]).T
        sigma_class /= class_counts[class_idx]
        within_class_scatter_matrix += fraction_classes[class_idx] * sigma_class
    return within_class_scatter_matrix

# 3.3
def get_between_class_scatter_matrix(seeds_dataset, mean_per_class):
    class_idxs, class_counts = np.unique(seeds_dataset[...,7], return_counts=True)
    fraction_classes = class_counts / class_counts.sum()
    mean_vector = seeds_dataset[...,:n_features].mean(axis=0)
    between_class_scatter_matrix = np.zeros((n_features, n_features))
    for class_idx in class_idxs: 
        class_idx = int(class_idx - 1)
        between_class_scatter_matrix += fraction_classes[class_idx] * (mean_per_class[class_idx] - mean_vector) * (mean_per_class[class_idx] - mean_vector).T
    return between_class_scatter_matrix

mean_per_class = get_mean_per_class(seeds_dataset)
within_class_scatter_matrix = get_within_class_scatter_matrix(seeds_dataset, mean_per_class)
between_class_scatter_matrix = get_between_class_scatter_matrix(seeds_dataset, mean_per_class)
