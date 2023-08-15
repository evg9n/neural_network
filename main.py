import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import utils
from os import listdir, path


def load_dataset() -> (list, list):
    x_train = keras.utils.image_dataset_from_directory(
        'train',
        subset='training',
        seed=42,
        validation_split=0.1,
        batch_size=256,
        image_size=(100, 100)
    )

    v_train = keras.utils.image_dataset_from_directory(
        'train',
        subset='validation',
        seed=42,
        validation_split=0.1,
        batch_size=256,
        image_size=(100, 100)
    )

    class_name = x_train.class_names

    test_train = keras.utils.image_dataset_from_directory(
        'test',
        batch_size=256,
        image_size=(100, 100)
    )

    return x_train, v_train, test_train


def create_neural_network():
    # Создаем последовательную модель
    model = Sequential()
    # Сверточный слой
    model.add(Conv2D(16, (5, 5), padding='same',
                     input_shape=(100, 100, 3), activation='relu'))
    # Слой подвыборки
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Сверточный слой
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    # Слой подвыборки
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Сверточный слой
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # Слой подвыборки
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Сверточный слой
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    # Слой подвыборки
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Полносвязная часть нейронной сети для классификации
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # Выходной слой, 131 нейрон по количеству классов
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    return model


def learning(model: Sequential, x_train, v_train, test_train):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    x_train = x_train.prefetch(buffer_size=AUTOTUNE)
    v_train = v_train.prefetch(buffer_size=AUTOTUNE)
    test_train = test_train.prefetch(buffer_size=AUTOTUNE)

    history = model.fit(x_train,
                        validation_data=v_train,
                        epochs=15,
                        verbose=2)

    scores = model.evaluate(test_train, verbose=1)
    print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))
    model.save('kamaz.h5')


def download_model() -> Sequential:
    model = keras.models.load_model('kamaz.h5')
    return model


def main():
    # x_train, v_train, test_train = load_dataset()
    # model = create_neural_network()
    # learning(model=model, x_train=x_train, v_train=v_train, test_train=test_train)

    model = download_model()
    for l in listdir('qwe'):
        img = keras.preprocessing.image.load_img(fr'qwe\{l}', target_size=(100, 100))
        x = keras.preprocessing.image.img_to_array(img=img)
        x = tf.expand_dims(x, axis=0)

        pred = model.predict(x)
        # print(pred)
        predicted_label = tf.argmax(pred[0])
        print(l)
        print('Predicted label:', predicted_label.numpy())


if __name__ == "__main__":
    main()
