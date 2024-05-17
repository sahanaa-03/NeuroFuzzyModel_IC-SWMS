import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define constants
input_shape = (150, 150, 3)
num_classes = 4
batch_size = 32
epochs = 20
validation_split = 0.2

# Define dataset directory
dataset_dir = r'C:\Users\sahan\Downloads\WDataset1'

# Load image paths and labels
image_paths = []
labels = []

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(class_name)  # Map class name to label

print(labels)

# Convert class names to integer labels
label_encoder = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_encoder[label] for label in labels])

# Split data into training and validation sets
train_paths, valid_paths, train_labels, valid_labels = train_test_split(image_paths, labels,
                                                                        test_size=validation_split,
                                                                       stratify=labels)

# Custom data generator
def custom_data_generator(image_paths, labels, batch_size):
    while True:
        batch_indices = np.random.randint(0, len(image_paths), batch_size)
        batch_paths = np.array(image_paths)[batch_indices]  # Convert to numpy array
        batch_labels = labels[batch_indices]

        batch_images = []
        for path in batch_paths:
            img = tf.keras.preprocessing.image.load_img(path, target_size=input_shape[:2])
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            batch_images.append(img_array)

        yield np.array(batch_images), np.array(batch_labels)

# Create data generators
train_generator = custom_data_generator(train_paths, train_labels, batch_size)
valid_generator = custom_data_generator(valid_paths, valid_labels, batch_size)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),  # Additional dense layer
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
print("Training the model...")
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))

    # Train the model for one epoch
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_paths) // batch_size,
        validation_data=valid_generator,
        validation_steps=len(valid_paths) // batch_size,
        epochs=1,
        verbose=1
    )

    # Print accuracy and loss after each epoch
    print("Training Accuracy:", history.history['accuracy'][0])
    print("Training Loss:", history.history['loss'][0])
    print("Validation Accuracy:", history.history['val_accuracy'][0])
    print("Validation Loss:", history.history['val_loss'][0])

# Save the trained model
model.save('trained_model3.h5')
