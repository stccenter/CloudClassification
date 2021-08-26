import os, sys
import numpy as np
import h5py
from scipy import io as scipyIO
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json, load_model
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from shutil import copyfile
import matplotlib.pyplot as plt
if __name__ == '__main__':
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
    estimator.fit(x_train, y_train, epochs=epoch,validation_data=(x_test,y_test), verbose=True, callbacks=[checkpoint])