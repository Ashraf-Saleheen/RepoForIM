
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activaotion, Maxpooling2D
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load()

x_train = x_train.reshape(-1, 28, 28, 1)
y_test = y_test.reshape(-1, 28, 28, 1)

def build_model():
    model = keras.model.sequential()

    model.add(Conv2D(32, (3,3), input_shape= x_train.shape[1:]))
    model.add(Activaotion("relu"))
    model.add(Maxpooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,(3,3)))
    model.add(Activaotion("relu"))
    model.add(Maxpooling2D(pool_size=(2,2)))

    model.add(Dense)
    model.add(Flatten(10))
    model.add(Activaotion("softmax"))

    model.compile(optimizer="adam",
                  loss = "sparse_Categorical_Crossentropy",
                   metrics= "[accuracy]" )
    return model
model = build_model()
model.fit([x_train], [y_train], batch_size=64, epoch = 5, validation_data=(x_test, y_test))





































