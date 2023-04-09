import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from yolov5.models import YoloV5
from yolov5.utils import freeze_all

# Define paths to data directories
train_dir = 'data/train'
val_dir = 'data/val'
output_dir = 'output'

# Set image size and batch size
IMG_SIZE = (416, 416)
BATCH_SIZE = 16

# Define data generators with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=45,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

# Define YOLOv5 model
yolo_model = YoloV5(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
yolo_model.summary()

# Freeze the backbone layers
freeze_all(yolo_model.backbone.layers)

# Define optimizer and compile the model
optimizer = Adam(lr=1e-4)
yolo_model.compile(optimizer=optimizer)

# Define callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=output_dir+'/weights.{epoch:02d}-{val_loss:.2f}.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Train the model
history = yolo_model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[checkpoint_callback]
)

# Save the final model weights
yolo_model.save_weights(output_dir+'/final_weights.h5')
