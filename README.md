# Cat vs. Dog Classifier Using Convolutional Neural Network (CNN)

This project implements a Convolutional Neural Network (CNN) to classify images into two categories: cats and dogs. The CNN is built using TensorFlow and Keras and includes data preprocessing, model building, training, and prediction.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Directory Structure](#directory-structure)
- [Data Preparation](#data-preparation)
- [Model Building](#model-building)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Usage](#usage)

## Prerequisites

Ensure you have the following libraries installed:

- TensorFlow
- Keras
- NumPy

You can install them using pip:

```bash
pip install tensorflow keras numpy
```

## Directory Structure
Organize your dataset and project files as follows:

```bash
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

- dataset/training_set/: Contains training images categorized into cats and dogs directories.
- dataset/test_set/: Contains test images categorized into cats and dogs directories.

## Data Preparation
The following code preprocesses the images from the dataset:

```python
from keras.preprocessing.image import ImageDataGenerator
```

# Preprocessing the Training set
```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
```

- rescale=1./255: Normalizes pixel values to the range [0, 1].
- shear_range, zoom_range, horizontal_flip: Data augmentation to make the model more robust.
- flow_from_directory: Loads and preprocesses images from the specified directory.

## Model Building
The CNN model is built as follows:

```python
import tensorflow as tf

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```
- Convolutional Layers: Extract features from the input images.
- Pooling Layers: Reduce the spatial dimensions.
- Flattening Layer: Converts the 2D matrices into 1D vectors.
- Fully Connected Layer: Provides a dense connection.
- Output Layer: Uses the sigmoid function to output binary predictions.
## Training the Model
Compile and train the model with the following code:
```python
# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
```

- Optimizer: Adam optimizer.
- Loss Function: Binary crossentropy.
- Metrics: Accuracy.
## Making Predictions
To make a prediction on a new image, use the following code:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess the image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict the class
result = cnn.predict(test_image)

# Decode the result
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
```

- image.load_img: Load and resize the image.
- image.img_to_array: Convert the image to a NumPy array.
- np.expand_dims: Add an extra dimension for batch processing.
- cnn.predict: Predict the class of the image.

## Saving and Loading the Model
### Saving the Model
After training your model, save it using:

```python
cnn.save('model_saved.keras')
```

cnn.save: Saves the entire model including its architecture, weights, and training configuration. The file model_saved.keras can be used to reload the model later without needing to retrain.

### Loading the Model
To load the saved model for future use, use:

```python
from tensorflow.keras.models import load_model
# Load the model
cnn = load_model('model_saved.keras')
```
load_model: Loads the saved model from the specified file.
## Usage
- Prepare your dataset and organize it according to the structure mentioned above.
- Run the model-building and training code to train your CNN.
- Save your trained model using the provided code snippet.
- Load the model for predictions using the corresponding code snippet.
- Run the prediction code to classify the image and print the result.
- Feel free to adapt and extend this code to fit your specific needs!
