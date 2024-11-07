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
    def __init__(self, n_qubits=1, Hamiltonian_c=None, alphas_c=None, embedding_code="rotation", sparsity_coefficient=0.95, L = 150):
        self.n_qubits = n_qubits
        self.length = L
        self.sparsity_coefficient = sparsity_coefficient

        self.Hamiltonian_c = Hamiltonian_c
        self.generate__H()

        self.hidden_alphas = alphas_c
        self.generate_alphas()

        self.embedding_code = embedding_code
        self.sampler = Sampler()

        self.K_train = None
        self.K_test = None



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

    def generate_alphas(self):
        if self.hidden_alphas is None:
            print("\nGenerating random alphas...")
            self.hidden_alphas = np.random.rand(self.length)
        alphas = np.maximum(self.hidden_alphas-self.sparsity_coefficient, 0)/(1-self.sparsity_coefficient)
        print(f"Non-zero alphas: {np.count_nonzero(alphas)}")
        
        while np.sum(alphas) == 0:
            print("All alphas are zero. Regenerating...")
            self.hidden_alphas /= max(self.hidden_alphas)
            alphas = np.maximum(self.hidden_alphas-self.sparsity_coefficient, 0)/(1-self.sparsity_coefficient)
            print(f"Non-zero alphas: {np.count_nonzero(alphas)}")

        alphas = alphas / np.sum(alphas)
        self.alphas = alphas

    def embedding(self, x, t):
        """Default data embedding method."""
        qc = QuantumCircuit(self.n_qubits)

        U = expm(-1j * self.H * 2 * np.pi * t / self.length)
        qc.unitary(Operator(U), range(self.n_qubits))

        if self.embedding_code == "euler":
            for i in range(self.n_qubits):
                qc.rx(x[t] * np.pi, i)
                qc.rz(x[t] * np.pi, i)
                qc.rx(x[t] * np.pi, i)

        elif self.embedding_code == "sincos":
            arcsin = np.arcsin((x[t]))
            arccos2 = np.arccos((x[t]**2))
            for i in range(self.n_qubits):
                qc.ry(arcsin, i)
                qc.rz(arccos2, i)
        else:
            raise ValueError(f"Unknown embedding code: {self.embedding_code}")    
        return qc

    def evaluate_similarity(self, x1, x2):
        """Evaluates the quantum kernel between two samples."""
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        assert len(x1) == len(x2), "Input vectors must have the same length."
        assert len(x1) == self.length, f"Input vectors must have length {self.length}."
        kernel_value = 0
        for t in range(self.length):
            if self.alphas[t] > 0:
                # Create the two quantum states |ψ⟩ and |φ⟩
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

                kernel_value += self.alphas[t] * inner_product
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