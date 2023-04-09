import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This code sets up ImageDataGenerators for both the training and validation sets, 
# applies image augmentation to the training set, sets the image size and batch size, 
# and defines the number of classes. The class indices and number of samples are then 
# printed for verification. This code can be used to preprocess data for both YOLO v5 
# and Faster R-CNN architectures.


# Define paths to data directories
train_dir = 'data/train'
val_dir = 'data/val'

# Set image size and batch size
IMG_SIZE = (416, 416)
BATCH_SIZE = 16

# Define data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=45,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define the number of classes
NUM_CLASSES = len(train_generator.class_indices)

# Print class indices
print('Class indices:', train_generator.class_indices)

# Print number of samples
print('Number of training samples:', train_generator.samples)
print('Number of validation samples:', val_generator.samples)


# Radar data is not necessarily image data, but it can be represented as image data. 
# Radar data typically consists of time-varying signals that are reflected from objects 
# in the environment. These signals can be processed to generate images that represent 
# the distribution of reflective surfaces within the radar's field of view. These images 
# are commonly referred to as radar images or radar maps.

# In the context of UAP tracking using radar data, the radar signals can be processed to 
# generate images that represent the UAP's location and trajectory, which can be used as 
# input to machine learning models such as YOLO v5 and Faster R-CNN. Therefore, in this 
# context, radar data can be considered as image data.
