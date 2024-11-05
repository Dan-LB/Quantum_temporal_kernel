from dtaidistance import dtw
import numpy as np
import itertools
from tqdm import tqdm


def calculate_dtw_similarity(series1, series2):
    distance, paths = dtw.warping_paths(series1, series2, use_c=False)
    best_path = dtw.best_path(paths)
    similarity_score = distance / len(best_path)
    return similarity_score

TYPES = {"TRAIN_KERNEL", "TEST_KERNEL", "ANOMALY_KERNEL"}

def generate_custom_kernel(DATASET1, DATASET2, TYPE, function, to_swap = True):
    if TYPE == "TRAIN_KERNEL":
        similarity_matrix = np.zeros((DATASET1.shape[0], DATASET1.shape[0]))
        train_combinations = list(itertools.combinations(range(DATASET1.shape[0]), 2))
        for i, j in tqdm(train_combinations):
        # series 1 is the first row
            X1 = DATASET1[i]
            # series 2 is the second row
            X2 = DATASET2[j]
            # Ensure both series are the same length before applying DTW
            assert len(X1) == len(X2)

            similarity_score = function(X1, X2)
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score  # Symmetric matrix

        if to_swap:
            similarity_matrix = 1 - (similarity_matrix / np.max(similarity_matrix))
        np.fill_diagonal(similarity_matrix, 1)

    elif TYPE == "TEST_KERNEL" or TYPE == "ANOMALY_KERNEL":
        similarity_matrix = np.zeros((DATASET1.shape[0], DATASET2.shape[0]))
        test_combinations = list(itertools.product(range(DATASET1.shape[0]), range(DATASET2.shape[0])))
        for i, j in tqdm(test_combinations):
            X1 = DATASET1[i]
            X2 = DATASET2[j]
            similarity_score = function(X1, X2)
            similarity_matrix[i, j] = similarity_score
        if to_swap:
                similarity_matrix = 1 - (similarity_matrix / np.max(similarity_matrix))
    return similarity_matrix