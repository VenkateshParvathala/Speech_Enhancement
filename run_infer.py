import time
start1 = time.perf_counter()
import logging
import multiprocessing
import os, json
import argparse
from keras.models import load_model
import config as cfg
import feat_utils as fe
import numpy as np
from keras.models import load_model
import pickle
import librosa
from librosa import istft, stft, output
import keras.losses
from lassoloss import cep_loss
keras.losses.cep_loss = cep_loss

#### Model ###################
model = None
def init_worker():
    global model
    model=load_model("resnet128_CepLoss_lstm.h5")
def process_input_by_worker_process(mixed_x):
    output = model.predict(mixed_x.reshape(1,-1,257))
    return output           
def run_inference_in_process_pool(model_param_path, input_lists, num_process):
    # Initialize process pool    
    process_pool = multiprocessing.Pool(processes=num_process, initializer=init_worker, initargs=())
    # Feed inputs to process pool to do inference
    pool_output = process_pool.map(process_input_by_worker_process, input_lists)

    return np.array(pool_output).reshape(-1,257)
def generate_context_inp_2d_list(wavfile_path,scaler):
    """
        :return: list of frames of STFT for a given wavefile
    """
    start = time.perf_counter()
    mixed_audio,_ = librosa.load( wavfile_path, sr = 16000)     
    mixed_complex_x = fe.calc_sp(mixed_audio,mode='complex')
    mixed_x = fe.log_sp(np.abs(mixed_complex_x))
    stft_end_time = time.perf_counter()
    mixed_x=scaler.transform(mixed_x)
    scaler_transform_end_time=time.perf_counter()
    mixed_x_3d,mixed_x_2d_list=fe.mat_2d_to_3d(mixed_x, hop=1)
    prepare_3d_list_end_time=time.perf_counter()
    return mixed_x_2d_list,mixed_complex_x,[start,stft_end_time,scaler_transform_end_time,prepare_3d_list_end_time]
def main():
  print("Time taken for all imports:", (time.perf_counter()-start1))
  num_process = 4
  model_param_path = "resnet128_CepLoss_lstm.h5"
  input_directory_path = "Noisy_Files/example_5min.wav"
  scaler=pickle.load(open('scaler_akm.p','rb'))
  input_2d_stft_list,mixed_complex_x,timings=generate_context_inp_2d_list(input_directory_path,scaler)
  print("TIme taken for perform stft:",timings[1]-timings[0])
  print("TIme taken for perform scaling:",timings[2]-timings[1])
  print("TIme taken for perform 2d to 3d:",timings[3]-timings[2])
  infer_start_time=time.perf_counter()
  pred=run_inference_in_process_pool(model_param_path, input_2d_stft_list, num_process)
  infer_end_time=time.perf_counter()
  print("Time taken for inference:",(infer_end_time-infer_start_time))
  pred=scaler.inverse_transform(pred)
  inv_scaling_end_time=time.perf_counter()
  print("Time taken for inverse scaling:",(inv_scaling_end_time-infer_end_time))
  pred_sp = np.exp(pred)
  s=istft((pred_sp * np.exp(1j*np.angle(mixed_complex_x))).T, win_length=cfg.win_length, hop_length=cfg.hop_length)
  enh_file='enh_wav_file.wav'
  output.write_wav(enh_file, s/max(abs(s)), 16000)
  synthesised_end_time=time.perf_counter()
  print("Time taken for synthesizing the enhanced wave file:",(synthesised_end_time-inv_scaling_end_time))
  print("Total time taken :",synthesised_end_time-start1)
if __name__ == "__main__":
    main()
