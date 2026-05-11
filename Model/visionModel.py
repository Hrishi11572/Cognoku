import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

# ── 3. Build Model ────────────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),  # extra layer for printed digits
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ── 4. Train on printed digits first ─────────────────────────────────
print("\nTraining on printed digits...")
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# ── 5. Fine-tune on MNIST (optional) ─────────────────────────────────
print("\nFine-tuning on MNIST...")
model.fit(
    x_mnist, y_mnist,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

# ── 6. Save ───────────────────────────────────────────────────────────
model.save('digit_classifier_printed.keras')
print("\nModel saved as digit_classifier_printed.keras")