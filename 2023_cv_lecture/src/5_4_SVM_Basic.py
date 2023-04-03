import cv2
import numpy as np

# Generate some random data
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
labels = np.random.randint(0, 2, (25, 1)).astype(np.float32)
labels = labels.astype(np.int32)  # Convert labels to integer values

# Create an SVM classifier
svm = cv2.ml.SVM_create()

# Set the SVM type to C_SVC (multi-class classification with optimal hyperplanes)
svm.setType(cv2.ml.SVM_C_SVC)

# Set the kernel type to linear
svm.setKernel(cv2.ml.SVM_LINEAR)

# Set the termination criteria for the SVM training algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
svm.setTermCriteria(criteria)

# Train the SVM classifier
svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)

# Define some test data
testData = np.random.randint(0, 100, (5, 2)).astype(np.float32)

# Predict the labels for the test data
_, results = svm.predict(testData)

# Print the results
print("Test Data:\n", testData)
print("Results:\n", results)
