import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Reduce memory usage and increase speed
tf.config.set_visible_devices([], 'GPU')  # Use CPU for faster startup
tf.keras.mixed_precision.set_global_policy('float32')

# Load MNIST dataset (use smaller subset for faster training)
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Use 20% of training data for faster training
train_size = len(X_train) // 5
X_train = X_train[:train_size]
y_train = y_train[:train_size]

# Preprocess training data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create simplified CNN model that's still accurate
model = Sequential([
    # First Conv2D layer
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    
    # Second Conv2D layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Dense layers with dropout
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model with smaller batch size for faster iterations
print("\nTraining CNN model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3,  # Reduced epochs
    batch_size=32,
    verbose=1
)

def preprocess_image(image_path):
    """Fast image preprocessing"""
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(image_path).convert('L'))
        
        # Store original
        original = img.copy()
        
        # Simple thresholding (faster than adaptive)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Simple padding
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(binary.shape[1] - x, w + 2*pad)
            h = min(binary.shape[0] - y, h + 2*pad)
            
            digit = binary[y:y+h, x:x+w]
        else:
            digit = binary
        
        # Resize to 28x28 (MNIST size)
        digit = cv2.resize(digit, (28, 28))
        
        # Normalize
        digit_array = digit.astype('float32') / 255.0
        digit_array = digit_array.reshape(1, 28, 28, 1)
        
        return digit_array, original, digit
    
    except Exception as e:
        raise Exception(f"Error processing image: {e}")

def predict_digit(image_path):
    """Fast digit prediction"""
    try:
        # Process image
        digit_array, original, processed = preprocess_image(image_path)
        
        # Get prediction
        predictions = model.predict(digit_array, verbose=0)
        prediction = np.argmax(predictions[0])
        confidence = float(predictions[0][prediction] * 100)
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3_pred = [(idx, float(predictions[0][idx] * 100)) for idx in top3_idx]
        
        # Print results
        print(f"\nPredicted Digit: {prediction} (Confidence: {confidence:.2f}%)")
        print("Top 3 predictions:")
        for digit, conf in top3_pred:
            print(f"  Digit {digit}: {conf:.2f}%")
        
        # Quick visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Original
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')
        
        # Processed
        ax2.imshow(processed, cmap='gray')
        ax2.set_title(f'Predicted: {prediction}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return prediction, confidence
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

if __name__ == "__main__":
    # Test with sample images
    test_images = ['Num.jpg', 'Number.jpg']
    for img_path in test_images:
        print(f"\nProcessing {img_path}...")
        predict_digit(img_path)
