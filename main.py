import os
import numpy as np
import cv2
import tensorflow as tf


def read_image():
    img = cv2.imread("image.jpg", 1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


#train the model
def train_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # image 784 pixels
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=10)
    tf.keras.models.save_model(model, savepath)


def load_model(savepath):
    return tf.keras.models.load_model(savepath)


def predict(img, model):
    img = np.expand_dims(img, axis=0)
    return model.predict(img)


def save_image():
    path = "E:\Program Files\programpython\pythonproject/image.jpg"
    img = cv2.imread(path, 0)
    cv2.imshow('image', img)
    cv2.imwrite('E:\Program Files\programpython\pythonproject/image2.png', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


#video displaying the test digit and their prediction finding by the model we trained
def save_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('E:\Program Files\programpython\pythonproject/output.avi', fourcc, 1, (28, 28), 0)
    count = 100
    for x in x_test:
        # cv2.imshow('frame', x)
        cv2.waitKey()
        x = np.uint8(255 * x)
        out.write(x)
        count -= 1
        if count == 0:
            break
    out.release()


if __name__ == '__main__':
    savepath = 'E:\Program Files\programpython\pythonproject/model.h5'
    x_train, y_train, x_test, y_test = load_mnist()
    train_model()
    load_model(savepath)
    save_video()
