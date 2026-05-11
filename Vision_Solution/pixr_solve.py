import pixrr as pix # type: ignore
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from PIL import Image

# Obtain an image 
test_image_path = "/Users/hrishikeshtiwari/Desktop/cognoku/test_files/test_img2.png"
img = pix.handle_image(test_image_path)

# Pre-processing of Image : smoothing, followed by adaptive thresholding
smoothed_img = pix.gaussian_smoothing(img)
thrs_img = pix.adaptive_thresh_gaussian(smoothed_img, inverse=True)

# Contour detection from the image 
contour = pix.contour_extractor(thrs_img)

# Code to Obtain individual cells 
height, width = contour.shape[:2]

cell_h = height // 9
cell_w = width // 9

margin_h = int(cell_h * 0.10)
margin_w = int(cell_w * 0.10)

cells = []

for r in range(9):

    row = []

    for c in range(9):

        y1 = r * cell_h
        y2 = (r + 1) * cell_h

        x1 = c * cell_w
        x2 = (c + 1) * cell_w

        cell = img[
            y1 + margin_h : y2 - margin_h,
            x1 + margin_w : x2 - margin_w
        ]

        row.append(cell)

    cells.append(row)


# Convert the cells to gray scale 
cells = np.array(cells)
gray_cells = []

for r in range(cells.shape[0]): 
    for c in range(cells.shape[1]):
        gray_cells.append(pix.iterative_global_thresholding(cells[r][c], inverse=True))

gray_cells = np.array(gray_cells)


# utility function to check if a cell actually contains an image 
def is_empty(roi, threshold=0.01):
    # Crop the outer border (the black frame around the cell)
    h, w = roi.shape[:2]
    margin_h = int(h * 0.20)
    margin_w = int(w * 0.20)
    cropped = roi[margin_h:h-margin_h, margin_w:w-margin_w]
    
    white_ratio = np.count_nonzero(cropped) / cropped.size
    return white_ratio < threshold


# Obtain the dataset 

# ── 1. Load your printed digits dataset ──────────────────────────────
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,          # augmentation for robustness
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    '/Users/hrishikeshtiwari/Desktop/cognoku/digits',             # your dataset path
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    '/Users/hrishikeshtiwari/Desktop/cognoku/digits',
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

print("Class indices:", train_gen.class_indices)  # verify 0-9 mapping

# ── 2. Mix with MNIST for extra robustness (optional) ────────────────
(x_mnist, y_mnist), _ = tf.keras.datasets.mnist.load_data()
x_mnist = x_mnist[..., np.newaxis] / 255.0  # (60000, 28, 28, 1)

# Sudoku does not use 0s, so filter them out to prevent false 0 predictions

x_mnist = x_mnist / 255.0  # normalize but don't add channel dim
mask = (y_mnist > 0)
X, y = x_mnist[mask], y_mnist[mask]

# Flatten (samples, 28, 28) → (samples, 784)
X = X.reshape(len(X), -1)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X, y)
print("KNN Model trained successfully.")


# Extracting the sudoku matrix 
final_matrix = []

for i in range(len(gray_cells)):
    raw_cell = gray_cells[i]
    if not is_empty(raw_cell, threshold=0.01):

        resized = np.array(Image.fromarray(raw_cell).resize((28, 28)))
        flattened = resized.reshape(1, -1) / 255.0  # (1, 784)

        prediction = knn.predict(flattened)
        digit = int(prediction[0])
        final_matrix.append(digit)
    else:
        final_matrix.append(0)

final_matrix = np.array(final_matrix).reshape((9, 9))
print(final_matrix) 


