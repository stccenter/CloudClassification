
import os, sys
import numpy as np
from scipy import io as scipyIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json, load_model
from keras.utils import np_utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from shutil import copyfile
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import argparse
import time
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide flag.')
    parser.add_argument("-f", "--flag", type=str, default='detection',
                        help="Specify the flag value: detection or rainy cloud")
    # Parse Arguments
    args = parser.parse_args()
    flag = args.flag
    SEED = 0
    np.random.seed(SEED)
    CLOUD_TRAIN_FOLDER = './train_10mm/'
    CLOUD_FLORENCE_FOLDER = './florence_10mm/'
    CLOUD_TEST_FOLDER = './test_10mm/'
    TIME = 'day'
    RAIN_CLOUD_FILENAME = TIME + '_rain_imerg.txt'
    NORAIN_CLOUD_FILENAME = TIME + '_norain_imerg.txt'  
    RAIN_RATE = 2.0 # adjust rate between the two type, rainy and non-rainy
    N_PREDICTOR = 15 #adjust predictor numbers
    INPUT_DIM = N_PREDICTOR

    def build_classifier():
        model = Sequential()
        model.add(Dense(units=512, input_dim=INPUT_DIM, kernel_initializer='normal', activation='relu'))
        model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(units=128, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def split_data(cloud_arr):
        x = cloud_arr[:, :N_PREDICTOR]
        labels = cloud_arr[:, -1]
        y = to_categorical(labels)
        return x, y, labels

    def load_data(folder, ratio):
        rain_cloud_path = folder + RAIN_CLOUD_FILENAME
        norain_cloud_path = folder + NORAIN_CLOUD_FILENAME

        rain_cloud_array = np.loadtxt(rain_cloud_path, delimiter=' ', dtype=np.float32, skiprows=0)
        norain_cloud_array = np.loadtxt(norain_cloud_path, delimiter=' ', dtype=np.float32, skiprows=0)

        rain_cloud_array = rain_cloud_array[~np.isnan(rain_cloud_array).any(axis=1)]
        norain_cloud_array = norain_cloud_array[~np.isnan(norain_cloud_array).any(axis=1)]

        rain_size = rain_cloud_array.shape[0]
        norain_size = norain_cloud_array.shape[0]

        smaller_size = min(rain_size, norain_size)
        new_rain_size = int(smaller_size // RAIN_RATE)
        new_norain_size = smaller_size

        random_rain_indices = np.random.choice(rain_size, size=new_rain_size, replace=False)
        random_norain_indices = np.random.choice(norain_size, size=new_norain_size, replace=False)
        new_rain_cloud_array = rain_cloud_array[random_rain_indices, :]
        new_norain_cloud_array = norain_cloud_array[random_norain_indices, :]

        cloud_array = np.concatenate((new_rain_cloud_array, new_norain_cloud_array), axis=0)
        cloud_array = shuffle(cloud_array)
        return cloud_array



    if flag == 'detection':
        data_TRAINING_file = './cloud_pc.sav'
        model_dir   = './detection_model'
        model_name  = 'clearsky'
        npc  = 75
        epoch =20

        data = scipyIO.readsav(data_TRAINING_file)
        predictors = np.array(data['X']).astype(np.float)
        predictands = np.array(data['Y']).astype(np.int32)
        dummy_predictands = np_utils.to_categorical(predictands)
        output_file = os.path.join(model_dir, model_name+'.h5')

        seed = 6
        x_train, x_test, y_train, y_test = train_test_split(predictors, dummy_predictands, test_size=0.3, random_state=seed)
        loss_str = 'categorical_crossentropy'
        optimizer_str = 'adam'
        def generate_kerasClassifier_model():
            model = Sequential()
            model.add(Dense(units=512,   input_dim=predictors.shape[1], activation='relu'))
            model.add(Dense(units=1024, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=2, activation='softmax'))
            model.compile(loss=loss_str, optimizer=optimizer_str, metrics=['accuracy'])
            return model
        estimator = KerasClassifier(build_fn=generate_kerasClassifier_model)
        checkpoint = ModelCheckpoint(output_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
        estimator.fit(x_train, y_train,  epochs=epoch,
                            validation_data=(x_test,y_test), verbose=True, callbacks=[checkpoint])
    if flag == 'rainy cloud':
        start_time = time.time()
        cloud_train = load_data(CLOUD_TRAIN_FOLDER, ratio=RAIN_RATE)
        cloud_florence = load_data(CLOUD_FLORENCE_FOLDER, ratio=RAIN_RATE)
        cloud_train_all = np.concatenate((cloud_train, cloud_florence), axis=0)
        cloud_train_all = np.repeat(cloud_train_all, repeats=6, axis=0)
        cloud_test = load_data(CLOUD_TEST_FOLDER, ratio=RAIN_RATE)
        x_org, y_org, labels_org = split_data(cloud_train_all)
        x_test, y_test, labels_test = split_data(cloud_test)
        scaler = MinMaxScaler()
        x_org = scaler.fit_transform(x_org)
        x_test = scaler.transform(x_test)
        x_train, x_val, y_train, y_val = train_test_split(
            x_org, 
            y_org,
            test_size=0.2, 
            random_state=SEED
        )
        print(len(x_train))
        estimator = KerasClassifier(build_fn=build_classifier)
        logdir = './logs/'
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        LOGS = logdir + datetime.now().strftime("%Y%m%d-%H%M%S")

        tboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir = LOGS,
            histogram_freq = 1,
            profile_batch = (2, 8)
        )
        EPOCHS = 100
        history = estimator.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=128,
            validation_data=(x_val, y_val), 
            verbose=True,
            callbacks = [tboard_callback]
        )
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        accuracy_score = estimator.score(x_test, y_test)
        print("Accuracy: {0:f}".format(accuracy_score))
        predictions = estimator.predict(x_test)
        pred_test = list(predictions)
        pred_test = np.array(pred_test)

        flag_a = np.logical_and(labels_test == 1, pred_test == 1)
        A = labels_test[flag_a]
        a = len(A)

        flag_b = np.logical_and(labels_test == 0, pred_test == 1)
        B = labels_test[flag_b]
        b = len(B)

        flag_c = np.logical_and(labels_test == 1, pred_test == 0)
        C = labels_test[flag_c]
        c = len(C)

        flag_d = np.logical_and(labels_test == 0, pred_test == 0)
        D = labels_test[flag_d]
        d = len(D)
        PDO = 1.0 * a / (a + c)
        POFD = 1.0 * b / (b + d)
        FAR = 1.0 * b / (a + b)
        Bias = 1.0 * (a + b) / (a + c)
        CSI = 1.0 * a / (a + b + c)
        AM = 1.0 * (a + d) / (a + b + c + d)
        CC = 1.0 * d / (b + d)
        print(a, b, c, d)
        print(PDO, POFD, FAR, Bias, CSI, AM)
        estimator.model.save('my_model')
        print("--- %s seconds ---" % (time.time() - start_time))
