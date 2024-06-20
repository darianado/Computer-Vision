""" 
Parking Occupancy Detection
----------------------------
Student Names: Dariana Dorin - 2163317
               Djibril Coulybaly - 2162985
Description of File: Using the trained MobileNet2 model, the program will 
                     apply the bounding boxes i.e. parking spaces to each frame of 
                     the video, and predict if the parking space is occupied or not. 
                     The occupancy will be highlighted (Green = Free, Red = Occupied)
                     and a count score will be monitored
"""  

# Importing essential libraries

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# Function to display a batch of images and their labels
def display_batch(images, labels, class_indices):
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):  # Display up to 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        label = np.argmax(labels[i])  # Get the index of the max value
        class_name = list(class_indices.keys())[
            list(class_indices.values()).index(label)
        ]
        plt.title(f"Label: {label} ({class_name})")
        plt.axis("off")
    plt.show()


# Define the target size
target_size = (32, 75)

epochs = 10


# Data generator with rescaling and validation split
data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

dataset_path = os.path.join("..","dataset")
train_ds = data_gen.flow_from_directory(
    directory=dataset_path,
    subset="training",
    seed=123,
    target_size=(32, 75),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)

val_ds = data_gen.flow_from_directory(
    directory=dataset_path,
    subset="validation",
    seed=123,
    target_size=(32, 75),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)


# Fetch a batch from the training dataset and display
train_batch = next(train_ds)
train_images, train_labels = train_batch
print(f"Training labels: {train_labels[:9]}")  # Print the first 9 labels
display_batch(train_images, train_labels, train_ds.class_indices)


# Define the MobileNetV2 model with pre-trained weights, without the top layers
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(32, 75, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces the spatial dimensions of the feature maps
predictions = Dense(2, activation="softmax")(x)  # Output layer with softmax activation

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Creating the models directory and saving the model
model_dir = os.path.join("..","models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, "model_mobilenetv2.keras"))

# Obtaining the validation loss and accuracy of the model
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
