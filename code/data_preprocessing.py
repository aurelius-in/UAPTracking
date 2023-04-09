import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to data directories
train_dir = 'data/train'
val_dir = 'data/val'

# Set image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Define the number of classes
NUM_CLASSES = 2

# Print class indices
print(train_generator.class_indices)

# Print number of samples in each class
print(train_generator.classes.shape)

# Print number of steps per epoch
steps_per_epoch = train_generator.samples // BATCH_SIZE
print(f'Steps per epoch: {steps_per_epoch}')

# Print number of validation steps
validation_steps = val_generator.samples // BATCH_SIZE
print(f'Validation steps: {validation_steps}')
