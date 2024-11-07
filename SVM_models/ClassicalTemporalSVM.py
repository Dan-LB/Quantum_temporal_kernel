import numpy as np
import itertools
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel

import matplotlib.pyplot as plt
from dtaidistance import dtw

# LIST ALL EMBEDDINGS HERE

MODELS = "DTW"


def calculate_dtw_similarity(series1, series2):
    distance, paths = dtw.warping_paths(series1, series2, use_c=False)
    best_path = dtw.best_path(paths)
    similarity_score = distance / len(best_path)
    return similarity_score

class ClassicalTemporalSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, model="DTW", L = 150):
        self.length = L
        self.model = model

        self.K_train = None
        self.K_test = None



        self.svc = None  # This will be our SVM model
        # If H or weights are not provided, set default values


    def evaluate_similarity(self, x1, x2):
        """Evaluates the quantum kernel between two samples."""
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        assert len(x1) == len(x2), "Input vectors must have the same length."
        assert len(x1) == self.length, f"Input vectors must have length {self.length}."
        kernel_value = 0
        if self.model == "DTW":
            kernel_value = calculate_dtw_similarity(x1, x2)
        else:
            raise ValueError(f"Unknown model: {self.model}")
        return kernel_value

    def build_kernel_matrix(self, DATASET1, DATASET2, TYPE):
        """Builds the kernel matrix using the quantum kernel, compatible with generate_custom_kernel."""
        if TYPE == "TRAIN_KERNEL":
            n_samples = DATASET1.shape[0]
            similarity_matrix = np.zeros((n_samples, n_samples))
            train_combinations = list(itertools.combinations(range(n_samples), 2))
            for i, j in tqdm(train_combinations, desc="Building TRAIN_KERNEL"):
                X1 = DATASET1[i]
                X2 = DATASET2[j]
                similarity_score = self.evaluate_similarity(X1, X2)
                similarity_matrix[i, j] = similarity_score
                similarity_matrix[j, i] = similarity_score  # Symmetric matrix
            # Fill diagonal entries
            for i in range(n_samples):
                similarity_matrix[i, i] = self.evaluate_similarity(DATASET1[i], DATASET1[i])

        elif TYPE == "TEST_KERNEL" or TYPE == "ANOMALY_KERNEL":
            n_samples_1 = DATASET1.shape[0]
            n_samples_2 = DATASET2.shape[0]
            similarity_matrix = np.zeros((n_samples_1, n_samples_2))
            test_combinations = list(itertools.product(range(n_samples_1), range(n_samples_2)))
            for i, j in tqdm(test_combinations, desc=f"Building {TYPE}"):
                X1 = DATASET1[i]
                X2 = DATASET2[j]
                similarity_score = self.evaluate_similarity(X1, X2)
                similarity_matrix[i, j] = similarity_score
        else:
            raise ValueError(f"Unknown TYPE: {TYPE}")
        
        similarity_matrix /= np.max(similarity_matrix)
        similarity_matrix = 1 - similarity_matrix
        return similarity_matrix

    def fit(self, X, y):
        """Fits the SVM model using the quantum kernel."""
        # Validate input
        X, y = check_X_y(X, y)
        # Scale features
        #X = self.scaler.fit_transform(X)
        self.X_ = X
        self.y_ = y
        # Build the kernel matrix
        K = self.build_kernel_matrix(X, X, TYPE="TRAIN_KERNEL")
        self.K_train = K
        # Train the SVM with the precomputed kernel
        self.svc = SVC(kernel='precomputed')
        self.svc.fit(K, y)
        return self


    def predict(self, X):
        """Predicts labels for new data."""
        check_is_fitted(self)
        # Validate input
        X = check_array(X)
        # Scale features
        #X = self.scaler.transform(X)
        # Build the kernel matrix between test data and training data
        K_test = self.build_kernel_matrix(X, self.X_, TYPE="TEST_KERNEL")
        self.K_test = K_test
        # Predict using the trained SVM
        return self.svc.predict(K_test)
    
    def predict_on_train(self):
        check_is_fitted(self)
        return self.svc.predict(self.K_train)
    

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def print_as_heatmap(self, kernel_matrix, title="Kernel Matrix"):
        """Prints the kernel matrix as a heatmap."""
        plt.imshow(kernel_matrix, cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.show()