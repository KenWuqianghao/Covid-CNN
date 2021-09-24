from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from glob import glob
import os, sys
# import necessary libraries and configurations
  
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

folders = glob(os.path.join(sys.path[0], "train/*"))
data_dir = os.path.join(sys.path[0], "train/")

batch_size = 32
image_height = 512
image_width = 1024
image_size = [image_height, image_width]

train_set = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size = (image_height, image_width),
  batch_size = batch_size)

val_set = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size = (image_height, image_width),
  batch_size = batch_size)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

train_set = train_set.map(lambda x, y: (normalization_layer(x), y))

val_set = val_set.map(lambda x, y: (normalization_layer(x), y))

inception = InceptionV3(input_shape=image_size + [3], weights='imagenet', include_top=False)

for layer in inception.layers:
    layer.trainable = False

x = Flatten()(inception.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=prediction)

opt = keras.optimizers.Adam(learning_rate=0.0002)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(
train_set,
validation_data = val_set,
epochs = 15,
verbose = 1
)

np.save(os.path.join(sys.path[0], "history.npy"),history.history)
model.save(os.path.join(sys.path[0], "trained_model.h5"))
model.save_weights(os.path.join(sys.path[0], "checkpoint"))