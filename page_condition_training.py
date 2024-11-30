'''import keras.src.losses
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def augmentation(n):
    for j in data_augmentation:
        n = j(n)
    return n


# Directories
train_data_directory = "Data/train"
test_data_directory = "Data/test"
data_directory = "DataV3"
model_file = "models/page_condition_v7.keras"

# Image size and batch size
image_size = (224, 224)
batch_size = 32

# Split the data into 90% training and 10% validation
train_data = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=0.3,  # 10% for validation
    subset="training",     # Load the training set
    seed=123,              # Seed for reproducibility
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=0.3,  # 10% for validation
    subset="validation",   # Load the validation set
    seed=123,              # Seed for reproducibility
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Data augmentation layers
data_augmentation = [
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
]

train_data = train_data.map(lambda x, y: (augmentation(x), y))

# Add rescaling layer separately
rescale_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Pre-trained model (MobileNet)
pre_trained_model = MobileNet(input_shape=[224, 224] + [3], weights="imagenet", include_top=False)

# Freeze pre-trained layers
for layer in pre_trained_model.layers:
    layer.trainable = False

# Model architecture
inputs = tf.keras.Input(shape=(224, 224, 3))

# Apply rescaling and data augmentation directly to the inputs
x = rescale_layer(inputs)

# Pass through pre-trained MobileNet
x = pre_trained_model(x, training=False)

# Compression and flattening
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)

# Add dropout
# x = keras.layers.Dropout(0.2)(x)

# Output layer
outputs = Dense(2, activation='softmax')(x)

# Create the model
new_model = Model(inputs, outputs)

# Compile the model
new_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ModelCheckpoint(model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
]

# Train the model
r = new_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=50,
    callbacks=callbacks
)'''


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

train_data_directory = "DataV4/train"
test_data_directory = "DataV4/test"
model_file = "models/page_condition_v9.keras"

train_augmentation = ImageDataGenerator(
    rescale=1.0/255,
    #rotation_range=20,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #fill_mode='nearest'
)

validation_augmentation = ImageDataGenerator(
    rescale=1.0/255
)

training_data = train_augmentation.flow_from_directory(
    train_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = validation_augmentation.flow_from_directory(
    test_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

pre_trained_model = MobileNet(input_shape=[224, 224] + [3], weights="imagenet", include_top=False)
# print(model.summary())


for layer in pre_trained_model.layers:
    layer.trainable = False

# Compress model layers using global average pooling
compressed_layer = tf.keras.layers.GlobalAveragePooling2D()(pre_trained_model.output)

# Flatten the compressed layer
temp_layer = Flatten()(compressed_layer)

# Define new output layer
new_out_layer = Dense(2, activation='softmax')(temp_layer)

new_model = Model(inputs=pre_trained_model.input, outputs=new_out_layer)

# Compile the new model
new_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# Callbacks to prevent overfitting
callbacks = [
    ModelCheckpoint(model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=15, verbose=1)
]

# Start training process
r = new_model.fit(
    training_data,
    validation_data=test_data,
    epochs=50,
    steps_per_epoch=len(training_data),
    validation_steps=len(test_data),
    callbacks=callbacks
)

