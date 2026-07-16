import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# SETTINGS
DATASET_PATH ="dogs-vs-cats/train/train"
IMAGE_SIZE = 64
SAMPLES_PER_CLASS = 1000
RANDOM_STATE = 42


def load_images():
    images = []
    labels = []

    cat_files = [f for f in os.listdir(DATASET_PATH) if f.startswith("cat.")]
    dog_files = [f for f in os.listdir(DATASET_PATH) if f.startswith("dog.")]

    random.seed(RANDOM_STATE)
    random.shuffle(cat_files)
    random.shuffle(dog_files)

    cat_files = cat_files[:SAMPLES_PER_CLASS]
    dog_files = dog_files[:SAMPLES_PER_CLASS]

    print("Loading cat images...")
    for filename in cat_files:
        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(image.flatten())
            labels.append(0)  # 0 = Cat

    print("Loading dog images...")
    for filename in dog_files:
        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(image.flatten())
            labels.append(1)  # 1 = Dog

    return np.array(images), np.array(labels)


# Load and prepare data
X, y = load_images()

# Convert pixels from 0–255 into 0–1
X = X / 255.0

print(f"\nTotal images loaded: {len(X)}")

# Divide images: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Train the SVM model
print("\nTraining SVM model. Please wait...")
model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["Cat", "Dog"]))

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Cat", "Dog"],
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.show()

# Show a few predictions
plt.figure(figsize=(12, 8))

for i in range(12):
    plt.subplot(3, 4, i + 1)

    image = X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
    actual = "Cat" if y_test[i] == 0 else "Dog"
    predicted = "Cat" if predictions[i] == 0 else "Dog"

    plt.imshow(image, cmap="gray")
    plt.title(f"Actual: {actual}\nPredicted: {predicted}")
    plt.axis("off")

plt.tight_layout()
plt.show()