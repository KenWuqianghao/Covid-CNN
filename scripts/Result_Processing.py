from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, sys

# import necessary libraries and configurations

model = load_model(os.path.join(sys.path[0], "trained_model.h5"))
model.load_weights(os.path.join(sys.path[0], "checkpoint"))
# load model for testing purposes

data_dir = os.path.join(sys.path[0], "test/")

batch_size = 32
image_height = 512
image_width = 1024  
image_size = [image_height, image_width]

validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
      directory = data_dir,
      classes = ['Healthy', 'Sick'],
      target_size=(image_height,image_width), 
      batch_size=batch_size,
      class_mode='binary',
      shuffle=False)

STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()

test_predict = model.predict(validation_generator, verbose = 1)
test_predict = np.array(test_predict)[:,1]

np.save(os.path.join(sys.path[0], "test_predict.npy"),test_predict)
np.save(os.path.join(sys.path[0], "test_labels.npy"),validation_generator.classes)