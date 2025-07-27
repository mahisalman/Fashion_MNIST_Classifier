
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import random
import pandas as pd

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
                                     Flatten, Dense, Dropout, BatchNormalization)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# For NN: Flattened input
x_train_nn = x_train.reshape(-1, 28*28)
x_test_nn = x_test.reshape(-1, 28*28)

# For CNN: Reshape to 28x28x1
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train_flat = x_train.reshape(-1, 784) / 255.0
x_test_flat = x_test.reshape(-1, 784) / 255.0

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype('float32')



# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("--- Building and Training Convolutional Neural Network (CNN) with Callbacks ---")

cnn_model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])


cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("\nCNN Model Architecture:")
cnn_model.summary()


early_stopping_cnn = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


checkpoint_filepath_cnn = 'best_cnn_model.keras'
model_checkpoint_cnn = ModelCheckpoint(
    filepath=checkpoint_filepath_cnn,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("\nTraining CNN Model with EarlyStopping and ModelCheckpoint...")
cnn_history = cnn_model.fit(x_train_cnn, y_train,
                            epochs=50,
                            validation_data=(x_test_cnn, y_test),
                            callbacks=[early_stopping_cnn, model_checkpoint_cnn])

print("\nCNN Model training complete.")

def plot_predictions(model, x_data, is_cnn=True, n=5):
    preds = model.predict(x_data[:n])
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        
        # Reshape if input is flattened (for NN)
        if is_cnn:
            img = x_data[i].reshape(28, 28)
        else:
            img = x_data[i].reshape(28, 28)

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[y_test[i]]}")
    
    plt.tight_layout()
    plt.show()


cnn_model.save("best_cnn_model.keras")

# 🔍 Display predictions for NN and CNN

print("🔍 Sample Predictions: CNN")
plot_predictions(cnn_model, x_test_cnn, is_cnn=True)

test_loss, test_acc = cnn_model.evaluate(x_test_cnn, y_test)
print(f"Test accuracy: {test_acc:.2%}")
