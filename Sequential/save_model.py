""" 
Parking Occupancy Detection
----------------------------
Student Names: Dariana Dorin - 2163317
               Djibril Coulybaly - 2162985
Description of File: Using the trained Keras Sequential model, the program will 
                     apply the bounding boxes i.e. parking spaces to each frame of 
                     the video, and predict if the parking space is occupied or not. 
                     The occupancy will be highlighted (Green = Free, Red = Occupied)
                     and a count score will be monitored
"""

# Importing essential libraries
import os
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input


# Function to display a batch of images and their labels
def display_batch(images, labels, class_indices):
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):  # Display up to 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(
            f"Label: {labels[i]} ({list(class_indices.keys())[list(class_indices.values()).index(labels[i])]})"
        )
        plt.axis("off")
    plt.show()


epochs = 10

data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

dataset_path = os.path.join("..","dataset")
train_ds = data_gen.flow_from_directory(
    directory=dataset_path,
    subset="training",
    seed=123,
    target_size=(29, 68),
    batch_size=32,
    class_mode="sparse",
    shuffle=True,
)

val_ds = data_gen.flow_from_directory(
    directory=dataset_path,
    subset="validation",
    seed=123,
    target_size=(29, 68),
    batch_size=32,
    class_mode="sparse",
    shuffle=True,
)


# Fetch a batch from the training dataset and display
train_batch = next(train_ds)
train_images, train_labels = train_batch
print(f"Training labels: {train_labels[:9]}")  # Print the first 9 labels
display_batch(train_images, train_labels, train_ds.class_indices)


# Define the Keras Sequential model
model = Sequential(
    [
        Input(shape=(29, 68, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training the model
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Creating the models directory and saving the model
model_dir = os.path.join("..","models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, "model.keras"))

# Obtaining the validation loss and accuracy of the model
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
