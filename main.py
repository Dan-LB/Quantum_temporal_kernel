import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
import itertools
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import numpy as np

import random

from SVM_models.models import evaluate_binary_SVM_standard, evaluate_binary_SVM_custom

# import GunPoint dataset from GunPoint/GunPoint_test.txt and GunPoint/GunPoint_train.txt


def load_dataset(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    # Parse each line, separating the label and features
    labels = []
    features = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        labels.append(int(values[0])-1)  # First value is the label, cast to int
        features.append(values[1:])    # Remaining values are features
    # Convert lists to numpy arrays for easier handling
    labels = np.array(labels)
    features = np.array(features)
    # normalize dataset to 0 1
    features = (features - features.min()) / (features.max() - features.min())
    return labels, features

# Load the GunPoint dataset
train_labels, train_features = load_dataset("GunPoint/GunPoint_train.txt")
test_labels, test_features = load_dataset("GunPoint/GunPoint_test.txt")


num_samples = 10

# Find indices of the first 10 samples for each class
class_1_indices = [i for i, label in enumerate(train_labels) if label == 0][:num_samples]
class_2_indices = [i for i, label in enumerate(train_labels) if label == 1][:num_samples]

# Plot samples for Class 1
plt.figure(figsize=(12, 6))
for idx in class_1_indices:
    plt.plot(train_features[idx], label="Class 1" if idx == class_1_indices[0] else "", color="blue")

# Plot samples for Class 2
for idx in class_2_indices:
    plt.plot(train_features[idx], label="Class 2" if idx == class_2_indices[0] else "", color="red")

# Add legend and title
plt.legend()
plt.title("First 10 Samples of GunPoint Dataset by Class")
plt.xlabel("Time")
plt.ylabel("Feature Value")
#plt.show()

KERNELS = {"linear", "poly", "rbf", "sigmoid"}

for kernel in KERNELS:
    print(f"Evaluating SVM with kernel: {kernel}")
    results = evaluate_binary_SVM_standard(train_features, test_features, train_labels, test_labels, kernel_code=kernel, print_results=True)

from kernels import calculate_dtw_similarity, generate_custom_kernel

K_train = generate_custom_kernel(train_features, train_features, "TRAIN_KERNEL", calculate_dtw_similarity)
K_test = generate_custom_kernel(test_features, train_features, "TEST_KERNEL", calculate_dtw_similarity)

from SVM_models.models import generate_kernels_dict

kernels_dict = generate_kernels_dict(K_train, K_test, None)

results = evaluate_binary_SVM_custom(kernels_dict, train_labels, test_labels, print_results=True)
