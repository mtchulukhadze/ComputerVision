import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.src.applications.resnet import ResNet50
from keras.src.layers import Dense, Flatten
from keras.src.models import Sequential

model  = Sequential()
base_model = ResNet50(include_top=False, weights='imagenet')
for layers in base_model.layers:
    layers.trainable = False
model.add(base_model)
model.add(Flatten())
model.add(Dense(units=60, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

X=[]
y=[]

animal_index = {}
MainDirectory ="PetImages"
for index,name in enumerate(os.listdir("PetImages")):
    animal_index[index] =name
print(animal_index)
for index, subfolder in enumerate(os.listdir(MainDirectory)):
    subpath =os.path.join(MainDirectory,subfolder)
    for file in os.listdir(subpath):
        image =cv2.imread(os.path.join(subpath,file))
        image =cv2.resize(image,(224,224))
        X.append(image)
        y.append(index)
X =np.array(X)
y =np.array(y)
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1)
X_train =X_train.astype('float32') /255.0
X_test =X_test.astype('float32') /255.0

model.fit(X_train, y_train, epochs=8, batch_size=70, validation_data=(X_test, y_test), verbose=1)

image1 = cv2.imread('download.jpg')
image1 = cv2.resize(image1, (224, 224))
image1 = np.expand_dims(image1, axis=0)
print(model.predict(image1))