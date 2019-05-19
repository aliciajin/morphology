import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import Image, display
import numpy as np
import json
import argparse
import datetime
import random

class classifier(object):
    def __init__(self, work_dir, data_augument):
        self.work_dir = work_dir
        self.data_augument = data_augument
        self.data_load_resize()
        self.train_test_split()
        if data_augument:
            self.data_agumentation()
        print("data augment: ", data_augument)
        self.final()


    def data_load_resize(self):
        labels = []
        images = []
        folders = ['Annotated_image_data_set_one_0-426/', 'Annotated_image_data_set_two_427_1003/']
        w_max, h_max = 0, 0
        for f in folders:
            data_csv = pd.read_csv(self.work_dir + f + "via_region_data.csv")
            data_csv = data_csv[data_csv['region_count'] == 1]
            data_csv = data_csv[data_csv['region_attributes'] != '{}']
            labels_curr = [json.loads(s) for s in data_csv['region_attributes'].tolist()]
            labels_curr = [d['Normal/Abnormal'] for d in labels_curr if len(d)>0]
            labels += labels_curr

            filenames = data_csv['#filename'].tolist()
            for file in filenames:
                # print(file)
                filepath = self.work_dir + f + file
                img_raw = tf.io.read_file(filepath)
                img_tensor = tf.image.decode_image(img_raw)
                images.append(img_tensor)
                h, w, _ = img_tensor.shape
                if h > h_max:
                    h_max = h
                if w > w_max:
                    w_max = w
        self.y = [0 if l == 'N' else 1 for l in labels]
        self.X = images
        # self.y = random.shuffle(self.y)
        # self.X = random.shuffle(self.X)
        # self.X = [tf.image.resize_with_pad(img_tensor, h_max, w_max) for img_tensor in images if img_tensor.shape != self.X[0].shape]
        self.w_max, self.h_max = w_max, h_max
        self.num_classes = len(np.unique(self.y))
        print('w_max, h_max: ', h_max, w_max)
        print('whole dataset with single object: ', len(self.y), len(self.X))

        print('input shape: ', self.X[0].shape)
        print('input shape: ', self.X[0].shape)
        print('classes: ', np.unique(self.y))
        print('Normal class: ', sum([1 for yi in self.y if yi == 0]))
        print('Abnormal class: ', sum([1 for yi in self.y if yi == 1]))

    def data_agumentation(self):
        print("process Data Augumentation on minority class: Normal class.")

        X_aug = []
        y_aug = []
        for x, y in zip(self.X_train, self.y_train):
            if y == 0:
                x_fliplr = np.fliplr(x)
                x_flipud = np.flipud(x)
                # x_flipud = tf.image.resize_with_pad(x_flipud, self.h_max, self.w_max)
                X_aug.append(x_fliplr)
                X_aug.append(x_flipud)
                y_aug.append(y)
                y_aug.append(y)

        # self.X_train = tf.concat(self.X_train, tf.convert_to_tensor(X_aug), 0)
        # self.y_train = tf.concat(self.y_train, tf.convert_to_tensor(y_aug), 0)
        self.X_train += X_aug
        self.y_train += y_aug
        print('Training size tripled after data augumentation: ', len(self.y_train), len(self.X_train))
        print('Normal class in training: ', sum([1 for yi in self.y_train if yi == 0]))
        print('Abnormal class in training: ', sum([1 for yi in self.y_train if yi == 1]))


    def train_test_split(self):
        X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test_tmp, y_test_tmp, test_size=0.5, random_state=231)
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        print("y_val: ", self.y_val)
        print("y_test: ", self.y_test)

    def final(self):
        self.X_train, self.X_val, self.X_test = tf.convert_to_tensor(self.X_train), tf.convert_to_tensor(self.X_val), tf.convert_to_tensor(self.X_test)
        self.y_train, self.y_val, self.y_test = tf.convert_to_tensor(self.y_train), tf.convert_to_tensor(self.y_val), tf.convert_to_tensor(self.y_test)
        print('X_train shape: ', self.X_train.shape)
        print('y_train shape: ', self.y_train.shape)
        print('X_val shape: ', self.X_val.shape)
        print('y_val shape: ', self.y_val.shape)
        print('X_test shape: ', self.X_test.shape)
        print('y_test shape: ', self.y_test.shape)

    def train(self, learning_rate=1e-3, batch_size=16, epochs=10):
        hidden_size = 100
        in_shape = self.X[0].shape

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=in_shape),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(32, (3, 3), input_shape=in_shape),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        # model = tf.keras.models.Sequential([
        #   tf.keras.layers.Flatten(input_shape=in_shape),
        #   tf.keras.layers.Dense(hidden_size, activation='relu'),
        #   # tf.keras.layers.Dropout(0.2),
        #   tf.keras.layers.Dense(self.num_classes, activation='softmax')
        # ])
        opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(self.X_val, self.y_val), verbose=2,
                    callbacks = [tensorboard_callback])
        model.evaluate(self.X_test, self.y_test, verbose=1)
        y_pred = model.predict_classes(self.X_test)
        print(precision_recall_fscore_support(self.y_test, y_pred, average='micro'))


parser = argparse.ArgumentParser()
parser.add_argument('--wd', dest='work_dir', default='/Users/apple/Documents/cs231n/project_v2/')
args = parser.parse_args()

my_work_dir = args.work_dir
# gcp_work_dir = '/home/apple/project/data/'  #'/Users/apple/Documents/cs231n/project_v2/'

clf = classifier(my_work_dir, data_augument = True)
clf.train()


