from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

import random

KERNELS = {"linear", "poly", "rbf", "sigmoid"}

def create_results_dict():
    return {
        "train_samples": [],
        "normal_samples": [],
        "anomaly_samples": [],
        "train": [],
        "test": [],
        "anomalies": []
    }

def evaluate_OneClassSVM_standard(train, test, anomaly, kernel_code, print_results=False):
    assert kernel_code in KERNELS, f"Invalid kernel code: {kernel_code}"

    one_class_svm = OneClassSVM(kernel=kernel_code, nu=0.1)
    one_class_svm.fit(train)
    test_predictions = one_class_svm.predict(test)
    test_labels = np.ones(test.shape[0])
    test_accuracy = accuracy_score(test_labels, test_predictions)

    anomaly_predictions = one_class_svm.predict(anomaly)
    anomaly_labels = -1 * np.ones(anomaly.shape[0])
    anomaly_accuracy = accuracy_score(anomaly_labels, anomaly_predictions)

    results_dict = create_results_dict()
    results_dict["train_samples"] = train.shape[0]
    results_dict["normal_samples"] = test.shape[0]
    results_dict["anomaly_samples"] = anomaly.shape[0]
    results_dict["train"] = test_accuracy
    results_dict["test"] = test_accuracy
    results_dict["anomalies"] = anomaly_accuracy

    if print_results:
        print(f"Train accuracy: {test_accuracy}")
        print(f"Test accuracy: {test_accuracy}")
        print(f"Anomalies accuracy: {anomaly_accuracy}")

    return results_dict

def generate_kernels_dict(K_train, K_test, K_anomaly):
    kernels_dict = {
        "TRAIN_KERNEL": K_train,
        "TEST_KERNEL": K_test,
        "ANOMALY_KERNEL": K_anomaly
    }
    return kernels_dict

def evaluate_OneClassSVM_custom(kernel_dict, print_results=False):
    train_kernel = kernel_dict["TRAIN_KERNEL"]
    test_kernel = kernel_dict["TEST_KERNEL"]
    anomaly_kernel = kernel_dict["ANOMALY_KERNEL"]

    one_class_svm = OneClassSVM(kernel="precomputed", nu=0.1)
    one_class_svm.fit(train_kernel)
    test_predictions = one_class_svm.predict(test_kernel)
    test_labels = np.ones(test_kernel.shape[0])
    test_accuracy = accuracy_score(test_labels, test_predictions)

    anomaly_predictions = one_class_svm.predict(anomaly_kernel)
    anomaly_labels = -1 * np.ones(anomaly_kernel.shape[0])
    anomaly_accuracy = accuracy_score(anomaly_labels, anomaly_predictions)

    results_dict = create_results_dict()
    results_dict["train_samples"] = train_kernel.shape[0]
    results_dict["normal_samples"] = test_kernel.shape[0]
    results_dict["anomaly_samples"] = anomaly_kernel.shape[0]
    results_dict["train"] = test_accuracy
    results_dict["test"] = test_accuracy
    results_dict["anomalies"] = anomaly_accuracy

    if print_results:
        print(f"Train accuracy: {test_accuracy}")
        print(f"Test accuracy: {test_accuracy}")
        print(f"Anomalies accuracy: {anomaly_accuracy}")

    return results_dict

# New function for binary classification with standard kernel
def evaluate_binary_SVM_standard(train, test, train_labels, test_labels, kernel_code, print_results=False):
    assert kernel_code in KERNELS, f"Invalid kernel code: {kernel_code}"

    binary_svm = SVC(kernel=kernel_code)
    binary_svm.fit(train, train_labels)
    train_predictions = binary_svm.predict(train)
    test_predictions = binary_svm.predict(test)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    if print_results:
        print(f"Train accuracy: {train_accuracy}")
        print(f"Test accuracy: {test_accuracy}")

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

# New function for binary classification with custom precomputed kernel
def evaluate_binary_SVM_custom(kernel_dict, train_labels, test_labels, print_results=False):
    train_kernel = kernel_dict["TRAIN_KERNEL"]
    test_kernel = kernel_dict["TEST_KERNEL"]

    binary_svm = SVC(kernel="precomputed")
    binary_svm.fit(train_kernel, train_labels)
    train_predictions = binary_svm.predict(train_kernel)
    test_predictions = binary_svm.predict(test_kernel)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    if print_results:
        print(f"Train accuracy: {train_accuracy}")
        print(f"Test accuracy: {test_accuracy}")

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }
