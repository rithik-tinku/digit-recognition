import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Add channel dimension
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)  # One-hot encode the labels
y_test = to_categorical(y_test, 10)

# Build a powerful CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(10, activation='softmax')  # Output layer for 10 digits
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the CNN model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to preprocess the input handwritten image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert the image if necessary
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img).astype('float32') / 255.0  # Normalize the image
    img = img.reshape(1, 28, 28, 1)  # Add batch dimension for model input
    return img

# Path to the handwritten digit image
image_path = 'digit1.jpg'

try:
    # Preprocess the input image
    processed_image = preprocess_image(image_path)

    # Predict the digit using the model
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Display the predicted result and the image
    print(f"Predicted Digit: {predicted_digit}")
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.show()

except Exception as e:
    print(f"Error processing the image: {e}")
