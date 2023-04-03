import cv2
import numpy as np

# Generate some random data
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
labels = np.random.randint(0, 2, (25, 1)).astype(np.float32)

# Create a K-Nearest Neighbors classifier
knn = cv2.ml.KNearest_create()

# Train the classifier
knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)

# Define some test data
testData = np.random.randint(0, 100, (5, 2)).astype(np.float32)

# Predict the labels for the test data
ret, results, neighbours, dist = knn.findNearest(testData, 3)

# Print the results
print("Test Data:\n", testData)
print("Results:\n", results)
print("Nearest Neighbours:\n", neighbours)
print("Distances:\n", dist)
