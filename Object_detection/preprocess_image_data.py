import os
import shutil
import imagenet_utils

# Download and extract the ImageNet dataset
imagenet_utils.download_imagenet_archive('path/to/save/dataset')

# Load the image paths and labels for the 'person' category
image_paths, labels = imagenet_utils.load_imagenet_subset('path/to/save/dataset', 'person')


# Set the path to the ImageNet dataset
data_dir = 'path/to/imagenet'

# Set the path to the train and validation directories
train_dir = 'path/to/train'
val_dir = 'path/to/val'

# Set the classes to include in the "person" category
person_classes = ['person', 'man', 'woman', 'boy', 'girl']

# Create the train and validation directories for the "person" category
for class_name in person_classes:
    # Create the class directories in the train directory
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    
    # Create the class directories in the validation directory
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    # Get the list of image files for the current class
    class_dir = os.path.join(data_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.JPEG')]
    
    # Split the image files into train and validation sets
    num_train = int(len(image_files) * 0.8)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]
    
    # Copy the train files to the train directory
    for filename in train_files:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(train_dir, class_name, filename)
        shutil.copy(src_path, dst_path)
    
    # Copy the validation files to the validation directory
    for filename in val_files:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(val_dir, class_name, filename)
        shutil.copy(src_path, dst_path)
