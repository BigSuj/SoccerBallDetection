import os
import random
from PIL import Image
import numpy as np
import time


def get_data(text_file_path, image_folder_path):
    # Create a dictionary to store the image paths and labels
    image_list = []

    # Read in the text file and populate the dictionary
    with open(text_file_path, "r") as file:
        for line in file:
            image_path, label = line.strip().split(" ")
            image_path = os.path.join(image_folder_path, image_path) + '.JPEG'
            image_list.append((image_path, int(label)))
    return image_list

def undersample_data(image_list):
    # Separate the images with and without a soccer ball
    ball_images = []
    noball_images = []

    for image_path, label in image_list:
        if label == 1:
            ball_images.append((image_path, 1))
        elif label == -1:
            noball_images.append((image_path, 0))

    # Determine the number of images to under-sample
    num_ball_images = len(ball_images)
    num_noball_images = len(noball_images)
    undersample_size = min(num_ball_images, num_noball_images)

    # Randomly select images from both classes
    sampled_ball_images = random.sample(ball_images, undersample_size)
    sampled_noball_images = random.sample(noball_images, undersample_size)

    # Combine the sampled images
    new_image_list = sampled_ball_images + sampled_noball_images

    # Shuffle the image paths
    random.shuffle(new_image_list)
    return new_image_list



def preprocess_images(image_list, batch_size, output_folder):
    # Define target image size
    img_size = (224, 224)

    # Create empty lists to store preprocessed images and their labels
    preprocessed_images = []
    labels = []

    # Loop through the image paths and preprocess each image
    for i, (image_path, label) in enumerate(image_list):
        # Load the image and resize it to the target size
        image = Image.open(image_path)
        image = image.resize(img_size)

        # Convert the image to RGB format and normalize its pixel values
        image = image.convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0

        # Add the preprocessed image and its label to the lists
        preprocessed_images.append(image)
        labels.append(label)

        # Check if we have reached the batch size or the end of the image list
        if len(preprocessed_images) == batch_size or i == len(image_list) - 1:
            # Convert the lists to numpy arrays
            preprocessed_images = np.array(preprocessed_images)
            labels = np.array(labels)

            # Append the preprocessed images and labels to the output file
            output_file = os.path.join(output_folder, "batch_{}.npz".format(i // batch_size))
            np.savez(output_file, images=preprocessed_images, labels=labels)

            # Clear the lists for the next batch
            preprocessed_images = []
            labels = []

        print('processing image {}'.format(i))

    return

def get_grass(cv_folder):
    grass_images = os.listdir(cv_folder)
    grass_images = [(os.path.join(cv_folder, file), 0) for file in grass_images]
    return grass_images
def get_ball(cv_folder):
    ball_images = os.listdir(cv_folder)
    ball_images = [(os.path.join(cv_folder, file), 1) for file in ball_images]
    return ball_images

start_time = time.time()
# Define paths to text file and image folder
text_file_path = "train_163.txt"
image_folder_path = "ILSVRC/Data/DET/train"
image_list = get_data(text_file_path, image_folder_path)
new_image_list = undersample_data(image_list)


cv_folder = 'images.cv_axvaa9mfe9svy7cliehmms/data/train/grass'
ball_folder = 'soccer_ball_folder/data/train/soccer_ball'
grass_images = get_grass(cv_folder)
ball_images = get_ball(ball_folder)
new_image_list += grass_images
new_image_list += ball_images


batch_size = 1000
output_folder = "preprocessed_data_batches"
output_file_prefix = "batch"
preprocess_images(new_image_list, batch_size, output_folder)





# Print the number of images in each class
num_ball_images = sum(1 for path, label in new_image_list if label == 1)
num_noball_images = sum(1 for path, label in new_image_list if label == 0)
print("Number of ball images:", num_ball_images)
print("Number of no-ball images:", num_noball_images)
end_time = time.time()
elapsed_time = end_time - start_time
print("Execution time:", elapsed_time, "seconds")