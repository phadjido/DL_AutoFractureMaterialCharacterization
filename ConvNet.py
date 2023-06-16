import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# user options
TrainSetDir = os.getcwd() + "/Data/UT"
ImgHeight = 256
ImgWidth = 256

# training set
TrainData = tf.keras.preprocessing.image_dataset_from_directory(
    TrainSetDir,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    batch_size=16,
    image_size=(ImgHeight, ImgWidth),
    shuffle=True,
    seed=123,
    validation_split=0.15,
    subset="training"
)

# validation set
ValidationData = tf.keras.preprocessing.image_dataset_from_directory(
    TrainSetDir,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    batch_size=16,
    image_size=(ImgHeight, ImgWidth),
    shuffle=True,
    seed=123,
    validation_split=0.15,
    subset="validation"
)

# efficient data management
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = TrainData.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = ValidationData.cache().prefetch(buffer_size=AUTOTUNE)

# dataset information
classNames = TrainData.class_names
print("class names: ", classNames)

for image_batch, labels_batch in TrainData.take(1):
    print("image size: ", image_batch.shape)
    print("batch size: ", labels_batch.shape)
    break

numChannels = image_batch.shape[3]
num_classes = 2

# model creation and training
input_shape = (ImgHeight, ImgWidth, numChannels)

model = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255,
                                                input_shape=input_shape),
    layers.Conv2D(filters=32,
                  kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(filters=32,
                  kernel_size=(3, 3),
                  activation='relu',
                  ),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(units=1, activation='sigmoid'),
    ])

model.summary()

model.compile(Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=50, verbose=2)
