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

import matplotlib.pyplot as plt

# LIST ALL EMBEDDINGS HERE
EMBEDDINGS = ["rotation", "sincos"]

I = np.array([[1, 0], [0, 1]])  # Identity
X = np.array([[0, 1], [1, 0]])  # Pauli-X
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y
Z = np.array([[1, 0], [0, -1]]) 

class QuantumSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=1, Hamiltonian_c=None, embedding_code="euler", L = 150):
        self.n_qubits = n_qubits

        self.Hamiltonian_c = Hamiltonian_c
        self.generate__H()

        self.embedding_code = embedding_code
        self.sampler = Sampler()

        # generate a dictionary with a list of K_trains [K_train1, K_train2, ..., K_trainL]
        self.length = L
        self.K_train_dict = {}
        self.K_test_dict = {}
        self.K_validation = {}

        self.hidden_alphas = np.ones(self.length)/self.length

        self.svc = None  # This will be our SVM model
        # If H or weights are not provided, set default values


    def generate__H(self):
        """Generates a random Hamiltonian for the given number of qubits by combining Pauli matrices."""
        pauli_matrices = [I, X, Y, Z]
        if self.Hamiltonian_c is None:
            print("\nGenerating random coefficients for the Hamiltonian...")
            coefficients = np.random.rand(4**self.n_qubits)
            self.Hamiltonian_c = coefficients
            
        coefficients = self.Hamiltonian_c
        H = None
        for i in range(4**self.n_qubits):
            indices = np.unravel_index(i, (4,) * self.n_qubits)
            kron_product = pauli_matrices[indices[0]]
            for idx in indices[1:]:
                kron_product = np.kron(kron_product, pauli_matrices[idx])
            kron_product = np.array(kron_product, dtype=complex)
            if H is None:
                H = coefficients[i] * kron_product
            else:
                H += coefficients[i] * kron_product
        self.H = H
        self.H = self.H / np.linalg.norm(self.H)  # Normalize the Hamiltonian

    def generate_K_train(self, X_train):
        n_samples = X_train.shape[0]
        train_combinations = list(itertools.combinations(range(n_samples), 2))
        for t in tqdm(range(self.length)):
            K_t = np.zeros((n_samples, n_samples))
            for i, j in train_combinations:
                x = X_train[i][t]
                y = X_train[j][t]
                similarity_score = self.evaluate_instant_similarity(x, y, t)
                K_t[i, j] = similarity_score
                K_t[j, i] = similarity_score
            for i in range(n_samples):
                x = X_train[i][t]
                K_t[i, i] = self.evaluate_instant_similarity(x, x, t)

            self.K_train_dict[t] = K_t
        print("K_train_dict generated")

    def generate_K_test(self, X_train, X_test):
        n_test = X_test.shape[0]
        n_train = X_train.shape[0]
        test_combinations = list(itertools.product(range(n_test), range(n_train)))
        for t in tqdm(range(self.length)):
            K_t = np.zeros((n_test, n_train))
            for i, j in test_combinations:
                x = X_test[i][t]
                y = X_train[j][t]
                similarity_score = self.evaluate_instant_similarity(x, y, t)
                K_t[i, j] = similarity_score
            self.K_test_dict[t] = K_t
        print("K_test_dict generated")

    def generate_K_validation(self, X_train, X_validation):
        n_validation = X_validation.shape[0]
        n_train = X_train.shape[0]
        validation_combinations = list(itertools.product(range(n_validation), range(n_train)))
        for t in tqdm(range(self.length)):
            K_t = np.zeros((n_validation, n_train))
            for i, j in validation_combinations:
                x = X_validation[i][t]
                y = X_train[j][t]
                similarity_score = self.evaluate_instant_similarity(x, y, t)
                K_t[i, j] = similarity_score
            self.K_validation[t] = K_t
        print("K_validation generated")


    def evaluate_instant_similarity(self, x1, x2, t):
        """Evaluates the quantum kernel between two samples."""

        psi = self.embedding(x1, t)
        phi = self.embedding(x2, t)
        # Inverse of psi
        psi_inv = psi.inverse()
        # Compose circuits
        phi.compose(psi_inv, inplace=True)
        # Measure
        qc_measured = phi.measure_all(inplace=False)
        # Execute
        job = self.sampler.run(qc_measured, shots=100)
        result = job.result()
        try:
            inner_product = result.quasi_dists[0][0]
        except:
            inner_product = 0
        return inner_product


    def embedding(self, x, t):
        """Default data embedding method."""
        qc = QuantumCircuit(self.n_qubits)

        U = expm(-1j * self.H * 2 * np.pi * t / self.length)
        qc.unitary(Operator(U), range(self.n_qubits))

        if self.embedding_code == "euler":
            for i in range(self.n_qubits):
                qc.rx(x * np.pi, i)
                qc.rz(x * np.pi, i)
                qc.rx(x * np.pi, i)

        elif self.embedding_code == "sincos":
            arcsin = np.arcsin((x))
            arccos2 = np.arccos((x**2))
            for i in range(self.n_qubits):
                qc.ry(arcsin, i)
                qc.rz(arccos2, i)
        else:
            raise ValueError(f"Unknown embedding code: {self.embedding_code}")    
        return qc

    def compute_kernels(self, X_train, X_test, X_validation=None):
        self.generate_K_train(X_train)
        self.generate_K_test(X_train, X_test)
        if X_validation is not None:
            self.generate_K_validation(X_train, X_validation)
       