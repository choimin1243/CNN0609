from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense,Dropout, MaxPooling2D,Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
import numpy as np
import os
import tensorflow as tf
from PIL import Image

import os,glob,sys,numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils



X_train, X_test, y_train, y_test=np.load("_image_data.npy",allow_pickle=True)

image_w=64
image_h=64

X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255
print(X_train.shape)
model=Sequential()
model.add(Conv2D(32,(3,3),padding="same",input_shape=X_train.shape[1:],activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model_dir='./model'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path=model_dir+"/human_classify.model"
checkpoint=ModelCheckpoint(filepath=model_path,monitor='val_loss',verbose=1,save_best_only=True)
early_stopping=EarlyStopping(monitor='val_loss',patience=7)

pixel=image_h*image_w*3
f=
img = Image.open(f)
img = img.convert("RGB")
img = img.resize((image_w, image_h))
data = np.asarray(img)

