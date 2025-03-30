
 from tensorflow.keras.optimizers import RMSprop
 from tensorflow.keras.preprocessing.image import ImageDataGenerator
 import cv2
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
 from tensorflow.keras.models import Model, load_model
 from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import f1_score
 from sklearn.utils import shuffle
 import imutils
 import numpy as np
 
 model = Sequential([
     Input(shape=(150, 150, 3)),
     Conv2D(100, (3, 3), activation='relu'),
     MaxPooling2D(2, 2),
 
     Conv2D(100, (3, 3), activation='relu'),
     MaxPooling2D(2, 2),
 
     Flatten(),
     Dropout(0.5),
     Dense(50, activation='relu'),
     Dense(2, activation='softmax')
 ])
 
 model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])