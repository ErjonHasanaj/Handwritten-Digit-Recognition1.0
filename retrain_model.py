import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import to_categorical

# Load original training data (optional, you might want to use only feedback data)
# (x_train, y_train), (_, _) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
# y_train = to_categorical(y_train, 10)

# Load feedback data
feedback_images = []
feedback_labels = []
feedback_folder = 'feedback_folder'  # Folder where feedback images are stored
for filename in os.listdir(feedback_folder):
    if filename.startswith('user_feedback_') and filename.endswith('.png'):
        label = int(filename.split('_')[-1].split('.')[0])
        img = load_img(f'{feedback_folder}/{filename}', color_mode='grayscale', target_size=(28, 28))
        img_array = img_to_array(img)
        feedback_images.append(img_array)
        feedback_labels.append(label)

# Convert feedback data to numpy arrays and preprocess
feedback_images = np.array(feedback_images).astype('float32') / 255.0
feedback_labels = to_categorical(feedback_labels, 10)

# Optionally combine feedback data with some original data
# x_combined = np.concatenate((x_train, feedback_images))
# y_combined = np.concatenate((y_train, feedback_labels))

# Or use only feedback data if it's sufficient
x_combined = feedback_images
y_combined = feedback_labels

# Load the existing model
model = load_model('mnist.h5')

# Retrain the model with combined data
model.fit(x_combined, y_combined, epochs=5, batch_size=128)

# Evaluate the model (optional)

# Save the updated model
model.save('mnist_updated.h5')
