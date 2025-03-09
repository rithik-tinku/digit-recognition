# Digit Recognition using CNN

A fast and accurate handwritten digit recognition system using Convolutional Neural Networks (CNN) implemented with TensorFlow/Keras. The model is specifically optimized for recognizing digits like 1 and 5 while maintaining quick processing times.

## Features

- CNN architecture with 2 Conv2D layers and MaxPooling
- Optimized preprocessing pipeline for handwritten digits
- Support for both .jpg and .png image formats
- Fast prediction with confidence scores
- Visual output showing original and processed images

## Requirements

- Python 3.11
- TensorFlow 2.13.0
- NumPy
- Matplotlib
- Pillow (PIL)
- OpenCV

## Installation

1. Create a virtual environment:
```bash
python -m venv venv311
```

2. Activate the virtual environment:
```bash
# Windows
.\venv311\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your digit image files in the project directory
2. Run the script:
```bash
python digits.py
```

The script will process test images (`Num.jpg` and `Number.jpg`) and display:
- Predicted digit with confidence score
- Top 3 predictions
- Visual comparison of original and processed images

## Model Architecture

- Input Layer: 28x28x1 (grayscale images)
- Conv2D Layer 1: 16 filters, 3x3 kernel, ReLU activation
- MaxPooling Layer 1: 2x2
- Conv2D Layer 2: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling Layer 2: 2x2
- Dense Layer: 64 units, ReLU activation
- Dropout Layer: 0.5 rate
- Output Layer: 10 units (digits 0-9), Softmax activation

## Performance

- Training using 20% of MNIST dataset for faster processing
- Validation accuracy during training
- Optimized for fast prediction while maintaining accuracy
- Special focus on digits 1 and 5 recognition
