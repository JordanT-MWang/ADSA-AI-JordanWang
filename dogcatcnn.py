#now cnn  
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("num pgu aval: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Define paths
base_path = '/home/jordanw7/koa_scratch/dogcattest/catdog-data'
train_path = os.path.join(base_path, 'train')
valid_path = os.path.join(base_path, 'valid')
test_path = os.path.join(base_path, 'test')
#putting the data into a way kares expects
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat','dog'], batch_size=10, shuffle = False)
assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2
imgs, labels = next(train_batches)

#model 
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2,2), strides=2),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2,2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
#predictions
test_imgs, test_labels = next(test_batches)
predictions = model.predict(x=test_batches, verbose=0)
import pandas as pd

# Get predicted class (0=cat, 1=dog)
pred_classes = np.argmax(predictions, axis=1)

# Get true class
true_classes = test_batches.classes

# Compare
correct = pred_classes == true_classes
accuracy = np.mean(correct)
print("Test accuracy:", accuracy)

# Save predictions to CSV
results = pd.DataFrame({
    "filename": test_batches.filenames,
    "true_class": true_classes,
    "pred_class": pred_classes,
    "correct": correct
})

# Make a folder to save results if it doesn’t exist
results_dir = os.path.join(base_path, "results")
os.makedirs(results_dir, exist_ok=True)

# Save CSV
results_file = os.path.join(results_dir, "predictions.csv")
results.to_csv(results_file, index=False)
print("Predictions saved to:", results_file)