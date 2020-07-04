#############################################################
#                                                           #
#   Author:  Dr K. Sri Rama Murty                           #
#   Co-Author:Sivaganesh Andhavarapu                        #
#   Institute: Indian Institute of Techonolgy Hyderabad     #
#                                                           #
#############################################################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Input, CuDNNLSTM, Bidirectional, CuDNNGRU, Add, Activation, Conv1D, Average
from keras.optimizers import Adam, SGD
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import config as cfg
import h5py
#from keras.utils import multi_gpu_model
from keras.initializers import RandomUniform
import tensorflow as tf
from keras.models import load_model
#from lassoloss import custom_loss
from keras.losses import mean_absolute_error as mae
from keras.losses import mean_squared_error as mse
import window as w
import numpy as np
import tensorflow
from tensorflow import spectral

def cep_loss(y_true,y_pred):
    y_pred=tf.cast(y_pred, tf.complex64)
    y_true=tf.cast(y_true, tf.complex64)
    
    print("-----   y_pred.shape -----",y_pred.shape)
    print("------- y_true.shape ------",y_true.shape)
    y_pred_cep=spectral.irfft(y_pred)
    y_true_cep=spectral.irfft(y_true)
    print("-----   y_true_cep.shape -----",y_true_cep.shape)
    print("-----   y_pred_cep.shape -----",y_pred_cep.shape)
    y_pred=tf.cast(y_pred, tf.float32)
    y_true=tf.cast(y_true, tf.float32)
    y_true_cep_win=w.win(y_true_cep[:,:257])
    y_pred_cep_win=w.win(y_pred_cep[:,:257])
    #y_true_cep_win=y_true_cep[:,:257]
    #y_pred_cep_win=y_pred_cep[:,:257]
    print("-----   y_true_cep_win.shape -----",y_true_cep_win.shape)
    print("-----   y_pred_cep_win.shape -----",y_pred_cep_win.shape)
    print("-----  y_true.shape -----",y_true.shape)
    print("-----  y_pred.shape -----",y_pred.shape)
    print("--------------------  0.0033*mse(y_true,y_pred) + 0.9967*mae(y_true_cep_win,y_pred_cep_win) -------------------------")
    
    return  0.0033*(mse(y_true,y_pred)) + 0.9967*mae(y_true_cep_win,y_pred_cep_win) 

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#feat_file='/hdd/speech_en/feat/feat_delay.h5'
#scaler_file='/data/sivaganesh/speech_enh/feat/scaler_akm.p'
feat_file='/data/sivaganesh/speech_enh/feat/feature_akm.h5'
fid=h5py.File(feat_file,'r');
X=fid['x']
Y=fid['y']
inputs=Input(shape=(cfg.n_context, cfg.n_freq))
x1, x1_h,_=LSTM(128,return_sequences=True, return_state=True)(inputs)
x2, x2_h,_=LSTM(128,return_sequences=True, return_state=True)(x1)
x1x2=Average()([x1,x2])
x3=LSTM(128,return_sequences=False, return_state=False)(x2)
x4=Average()([x1_h,x2_h,x3])
x5=Dense(1024,activation='tanh')(x3)
outputs=Dense(cfg.n_freq, activation='linear')(x5)
model=Model(inputs=inputs, outputs=outputs)
model.summary()
lr=0.0002
#model=model(model,gpus=4)
model.compile(loss=cep_loss, optimizer=Adam(lr=lr))
for i in range(10):
    model.fit(X, Y, batch_size=256, validation_split=0, shuffle=False, epochs=1)
    model.save('resnet128_CepLoss_lstm.h5')
