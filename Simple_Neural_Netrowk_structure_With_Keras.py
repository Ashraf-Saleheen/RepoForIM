import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten
from keras.models import Sequential

fashion_data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = fashion_data.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_model()
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
