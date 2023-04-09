import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Flatten, Dense

# Define paths to data directories
train_dir = 'data/train'
val_dir = 'data/val'
output_dir = 'output'

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

# Define the Faster R-CNN model
base_model = EfficientNetB0(include_top=False, input_shape=(None, None, 3))
inputs = Input(shape=(None, None, 3))
x = base_model(inputs)
x = Flatten()(x)
x = Dense(NUM_CLASSES + 4, activation='linear')(x)
model = tf.keras.Model(inputs, x)

# Compile the model with Adam optimizer
optimizer = Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss={'output_1': 'categorical_crossentropy', 'output_2': 'mse'}
)

# Define callbacks for saving model and TensorBoard visualization
checkpoint = ModelCheckpoint(output_dir+'/faster_rcnn_best_model.h5', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir=output_dir+'/logs')

# Train the model
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[checkpoint, tensorboard]
)

# Evaluate the trained model
evaluation = model.evaluate(val_generator)
print(f'Validation loss: {evaluation[0]}')
print(f'Validation accuracy: {evaluation[1]}')

# Save the trained model
model.save(output_dir+'/faster_rcnn_final_model.h5')

# Visualize the model architecture
plot_model(model, to_file=output_dir+'/faster_rcnn_architecture.png', show_shapes=True)

