import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import interpolate

# Load the trained CNN model
model = load_model('trained_model3.h5')

# Define input variables for fuzzy logic
recyclable_prob = np.arange(0, 1.1, 0.1)
non_recyclable_prob = np.arange(0, 1.1, 0.1)
organic_prob = np.arange(0, 1.1, 0.1)
hazardous_prob = np.arange(0, 1.1, 0.1)

# Define output variable for fuzzy logic
category = np.arange(0, 5.1, 1)

# Define membership functions
def trimf(x, params):
    """
    Triangular membership function.

    Args:
        x (float or array-like): Input values.
        params (list): Parameters of the triangular function (lower, middle, upper).

    Returns:
        numpy.array: Trimf values.
    """
    return np.maximum(0,
                      np.minimum((x - params[0]) / (params[1] - params[0]), (params[2] - x) / (params[2] - params[1])))


# Membership function parameters
recyclable_low = [0, 0, 0.7]
recyclable_high = [0.3, 1, 1]

non_recyclable_low = [0, 0, 0.7]
non_recyclable_high = [0.3, 1, 1]

organic_low = [0, 0, 0.7]
organic_high = [0.3, 1, 1]

hazardous_low = [0, 0, 0.7]
hazardous_high = [0.3, 1, 1]

recyclable_mf = interpolate.interp1d(recyclable_prob, trimf(recyclable_prob, recyclable_low))
non_recyclable_mf = interpolate.interp1d(non_recyclable_prob, trimf(non_recyclable_prob, non_recyclable_low))
organic_mf = interpolate.interp1d(organic_prob, trimf(organic_prob, organic_low))
hazardous_mf = interpolate.interp1d(hazardous_prob, trimf(hazardous_prob, hazardous_low))


# Fuzzy rules
def fuzzy_inference(recyclable_prob, non_recyclable_prob, organic_prob, hazardous_prob):
    """
    Apply fuzzy logic rules to infer waste categories.

    Args:
        recyclable_prob (float): Probability of being recyclable.
        non_recyclable_prob (float): Probability of being non-recyclable.
        organic_prob (float): Probability of being organic.
        hazardous_prob (float): Probability of being hazardous.

    Returns:
        str or None: Inferred waste category or None if no inference.
    """
    if abs(recyclable_prob - non_recyclable_prob) < 0.1:
        return "recyclable and non-recyclable"

    if abs(organic_prob - hazardous_prob) < 0.1:
        return "organic and hazardous"

    if abs(recyclable_prob - organic_prob) < 0.1:
        return "recyclable and organic"

    if abs(non_recyclable_prob - hazardous_prob) < 0.1:
        return "non-recyclable and hazardous"

    if abs(organic_prob - non_recyclable_prob) < 0.1:
        return "organic and non-recyclable"

    if abs(organic_prob - recyclable_prob) < 0.1:
        return "organic and recyclable"

    if abs(non_recyclable_prob - recyclable_prob) < 0.1:
        return "non-recyclable and recyclable"

    if abs(hazardous_prob - organic_prob) < 0.1:
        return "hazardous and organic"

    if abs(hazardous_prob - non_recyclable_prob):
        return "hazardous and non-recyclable"

    return None


# Function to classify the image using CNN model
def classify_image(image_path):
    """
    Classify the image using the trained CNN model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image, dict: Image object and probabilities of waste categories.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255

    predictions = model.predict(img_array)[0]
    probabilities = {category: prob for category, prob in
                     zip(["recyclable", "non-recyclable", "organic", "hazardous", "uncertain"], predictions)}

    return img, probabilities


# Function to open file dialog and classify the selected image
def open_image():
    """
    Open file dialog to select an image, classify it, and display the result.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img, probabilities = classify_image(file_path)

        # Display the chosen image
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Display classification result
        result_label.config(text="Classifying...")

        # Check if highest two probabilities are similar
        sorted_probs = sorted(probabilities.values(), reverse=True)
        if sorted_probs[0] - sorted_probs[1] < 0.1:  # Adjust the threshold as needed
            max_prob_indices = np.argsort(-np.array(list(probabilities.values())))[:2]
            max_categories = [list(probabilities.keys())[i] for i in max_prob_indices]
            result_label.config(
                text=f"Classified with fuzzy logic as there are two categories of waste in this picture:\n{max_categories}\n\nProbabilities:\n{probabilities}")
        else:
            max_category = max(probabilities, key=probabilities.get)
            result_label.config(text=f"Classified as:\n{max_category}\n\nProbabilities:\n{probabilities}")


# Create the Tkinter window
root = tk.Tk()
root.title("Waste Image Classifier")

# Create the main frame
main_frame = tk.Frame(root)
main_frame.pack(padx=20, pady=20)

# Create widgets
browse_button = tk.Button(main_frame, text="Browse Image", command=open_image)
browse_button.pack(pady=10)

image_label = tk.Label(main_frame)
image_label.pack(pady=10)

result_label = tk.Label(main_frame, text="")
result_label.pack(pady=10)

# Run the application
root.mainloop()
