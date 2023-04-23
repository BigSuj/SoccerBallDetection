import numpy as np
import os
from keras import layers, models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the batch size and number of classes
batch_size = 32
num_classes = 2

# Define the input shape of the model
input_shape = (224, 224, 3)
processed_dir = 'preprocessed_data_batches'
all_batches = os.listdir(processed_dir)

def load_data(processed_dir, current_batch, batch_num):
    all_batches = os.listdir(processed_dir)
    batches = all_batches[current_batch:batch_num]
    batches = [os.path.join(processed_dir, file) for file in batches]

    images = np.empty((0, 224, 224, 3))
    label = np.array([])
    for npz_file in batches:
        NPZ = np.load(npz_file)
        images = np.concatenate((images, NPZ['images']), axis=0)
        label = np.concatenate((label, NPZ['labels']))
        
    return images, label

def get_data(batch, batch_num, current_batch):
    images, labels = load_data(processed_dir, current_batch, batch_num)
    print('retreived training data, batch {}.'.format(batch))
    return images, labels

# Define the model architecture (simple convolutional neural network)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

# Compile the model (define the loss function, optimizer, and metrics)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print('compiled model.')

batch_num = 5
current_batch=0
count = 1
while True:
    images, labels = get_data(count, batch_num, current_batch)
    # Split the data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2)
    # convert labels to one hot:
    labels_one_hot = to_categorical(labels)

    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

    # Train the CNN model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    current_batch = batch_num
    batch_num += 5
    if batch_num > len(all_batches):
        print('completed.')
        break
    print('batch {} training complete.'.format(count))
    count += 1

# Save the trained model
model.save("cnn_model.h5")



