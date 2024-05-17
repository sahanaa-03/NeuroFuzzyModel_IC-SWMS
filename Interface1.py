import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the pre-trained CNN model
model = tf.keras.models.load_model('trained_model3.h5')

# Function to classify the selected image
def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Open the image using PIL
        img = Image.open(file_path)
        # Resize the image to match the input size of the model
        img = img.resize((150, 150))
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Normalize the image
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Perform classification
        prediction = model.predict(img_array)
        # Get the predicted class label
        class_index = np.argmax(prediction)
        classes = ["Recyclable", "Non-Recyclable", "Organic", "Hazardous"]
        class_label = classes[class_index]
        # Get the probability of the predicted class
        probability = prediction[0][class_index]

        # Display the result
        result_label.config(text=f"Classified as: {class_label}\nProbability: {probability:.2f}")

        # Display the selected image
        img = Image.open(file_path)
        img.thumbnail((200, 200))  # Resize image for display
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to the image to prevent garbage collection


# Create the Tkinter window
root = tk.Tk()
root.title("Image Classifier")

# Create the main frame
main_frame = tk.Frame(root)
main_frame.pack(padx=50, pady=50)

# Create widgets
browse_button = tk.Button(main_frame, text="Browse Image", command=classify_image)
browse_button.pack(pady=10)

image_label = tk.Label(main_frame)
image_label.pack(pady=10)

result_label = tk.Label(main_frame, text="")
result_label.pack(pady=10)

# Run the application
root.mainloop()
