""" 
Parking Occupancy Detection
----------------------------
Student Names:          Dariana Dorin - 2163317
                        Djibril Coulybaly - 2162985
                        
Description of File:    Using the trained SVC with GridSearchCV, the program will apply the bounding boxes i.e. parking spaces 
                        to each frame of the video, and predict if the parking space is occupied or not. The occupancy will be 
                        highlighted (Green = Free, Red = Occupied) and a count score will be monitored
"""

# Importing essential libraries
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib


# Function to load images and labels from the dataset folder
def load_images(folder, target_size=(68, 29)):
    images = []
    labels = []
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # Resize image to correct target size
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        images.append(img)
                        labels.append(label)
    return images, labels


# Function to extract HOG features from the images obtained in the dataset
def extract_hog_features(images):
    hog_features = []
    for image in images:
        feature = hog(
            image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False
        )
        hog_features.append(feature)
    return np.array(hog_features)


# Load dataset
dataset_folder = os.path.join("..","dataset")
images, labels = load_images(dataset_folder)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Fucntion call to extract HOG features
hog_features = extract_hog_features(images)

# Splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    hog_features, labels, test_size=0.3, random_state=42
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SVC with GridSearchCV                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Train the model
svc_params = {"kernel": ("linear", "rbf"), "C": [1, 10]}
svc = SVC(probability=True)
svc_clf = GridSearchCV(svc, svc_params)
svc_clf.fit(X_train, y_train)


# Predict and display the evaluation of the model
svc_best = svc_clf.best_estimator_
svc_predictions = svc_best.predict(X_test)
svc_probabilities = svc_best.predict_proba(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
svc_loss = log_loss(y_test, svc_probabilities)
print("SVC Accuracy:", svc_accuracy)
print("SVC Loss:", svc_loss)
print("SVC Classification Report:\n", classification_report(y_test, svc_predictions))

models_folder = os.path.join("..","models")
if not os.path.exists(models_folder):
    os.makedirs(models_folder)
joblib.dump(svc_best, os.path.join(models_folder, "svc_model.pkl"))
