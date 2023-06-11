import tensorflow as tf
from tensorflow import keras

fashion_data= tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = fashion_data.load_data()

x_train = x_train/255
x_test = x_test/255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss = "sparse_categorical_crossentropy",
              metrics= ["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test,y_test))

model.evaluate(x_test, y_test)
