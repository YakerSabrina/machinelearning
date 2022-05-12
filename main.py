import cv2
import os

import tensorflow as tf
import numpy as np

import keras

def load_image(path):
    img = cv2.imread(path, 1)
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

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

def save_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('E:\Program Files\programpython\pythonproject/output.avi', fourcc, 1, (28, 28), 0)
    count = 10
    for x in x_test:
        # cv2.imshow('frame', x)
        cv2.waitKey()
        x = np.uint8(255 * x)
        out.write(x)
        count -= 1
        if count == 0:
            break
    out.release()


def show_image_pred():
    model = load_model('E:\Program Files\programpython\pythonproject/model.h5')
    xtrain, y_train, xtest, y_test= load_mnist()
    cap = cv2.VideoCapture('E:\Program Files\programpython\pythonproject/output.avi', 0)
    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter('E:\Program Files\programpython\pythonproject/output1.avi', fourcc1, 1, (56, 56), 0)
    count = 0
    while (cap.isOpened()):
        #print count
        ret, frame = cap.read()
        if count == 10:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_gray = np.array(frame_gray, dtype='uint8')

        pred = predict(frame_gray, model)
        index = np.argmax(pred)

        img = np.zeros((56, 56), np.uint8)
        img_gray = np.array(img, dtype='uint8')
        img_gray[10:38, 0:28] = frame_gray

        cv2.putText(img_gray, str(index), (34, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255))

        out1.write(img_gray)
        count += 1
        cv2.imshow('img_gray', img_gray)
        #cv2.imwrite('test' + str(count) + ".jpeg", img_gray)
        cv2.waitKey()


    cap.release()
    out1.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    savepath = 'E:\Program Files\programpython\pythonproject/model.h5'
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(x_train.shape)
    save_video()
    train_model()
    show_image_pred()