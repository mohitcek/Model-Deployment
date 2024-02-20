import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 


# Define a simple sequential model
def create_model():
    model = Sequential()
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
    model.add(MaxPool2D())
    
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics=['accuracy'])

    return model
