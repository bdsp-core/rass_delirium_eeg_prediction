from collections import Counter
import numpy as np
import scipy.io as sio
import torch as th
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
from read_delirium_data import *
from segment_EEG import *
from rass_delirium_tracking.eegnet_feedforward import EEGNet_CNN, EEGNet_CNN_CAMICU
from rass_delirium_tracking.eegnet_recurrent import EEGNet_RNN, ModuleSequence
from rass_delirium_tracking.braindecode.torch_ext.util import np_to_var, var_to_np


if __name__=='__main__':

    K = 6  # currently we only track 6 levels in RASS (-5, -4, ..., 0) [DO NOT CHANGE]
    loss_function = 'ordinal'  # use ordinal regression [DO NOT CHANGE]
    use_gpu = True
    line_freq = 60.  # [Hz], in US it is 60Hz
    bandpass_freq = [0.5, 20.]  # [Hz] [DO NOT CHANGE]
    window_time = 4  # [s] [DO NOT CHANGE]
    window_step = 4  # [s] [DO NOT CHANGE]
    amplitude_thres = 500  # uV
    
    ## load data
    
    #TODO implement your own data loading function
    # make sure it returns a dict:
    # {'channel_names': a list of channel/electrode names, 'Fs': sampling frequency, a float, in Hz, 'data': 4 channels x #sampling points, numpy array, in *uV*}
    # the channels must be [Fp1, Fp2, F7, F8] (order matters), or nearby channels, since the model is only trained on these channels
    # for example:
    res = read_delirium_mat('/data/delirium/github_rass_delirium/example_data/M5060612_140120_2323_part2_converted.edf.mat')
    
    # preprocess and segment the signal
    newFs = 250.  # resample to 250Hz [DO NOT CHANGE]
    X, seg_start_ids, seg_mask, specs_db, freqs = segment_EEG(res['data'],
            window_time, window_step, res['Fs'], newFs,
            notch_freq=line_freq, bandpass_freq=bandpass_freq,
            amplitude_thres=amplitude_thres, n_jobs=-1)
    # X: EEG segments, (#segment, 4 channels, 1000 = 250Hz x 4s sampling points), numpy array
    # seg_start_ids: (#segment,), numpy array, the starting point id of each segment in the original signal res['data']
    #                note the point ids is after resampling to newFs=250Hz
    # seg_mask: the status of each segment
    # specs_db: spectrogram in dB
    # freqs: the frequency in specs_db
    print(Counter(seg_mask))
    
    # prepare data to feed into the model
    newFs = 62.5  # resample to 62.5Hz [DO NOT CHANGE]
    X = X[...,::4]  # [DO NOT CHANGE]
    # make the final size of input (#recording, #segment, #channel=2, #sampling points=62.5Hz x 4s=250)
    X = X[None,...]
    if use_gpu:
        X = X.astype('float32')
    X = np_to_var(X)
    if use_gpu:
        X = X.cuda()
    
    
    
    ## load the RASS model
    
    # works under Python 2.7 and PyTorch 0.4.0
    model_cnn_rass = EEGNet_CNN(loss_function, K)
    model_cnn_rass.load_state_dict(th.load('models/model_RASS_cnn.pth'))
    model_rnn1_rass = EEGNet_RNN('ordinal', 192, K, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=16)
    model_rnn1_rass.load_state_dict(th.load('models/model_RASS_rnn1.pth'))
    model_rnn2_rass = EEGNet_RNN('ordinal', 16, 6, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=4)
    model_rnn2_rass.load_state_dict(th.load('models/model_RASS_rnn2.pth'))
    model_rass = ModuleSequence([model_cnn_rass, model_rnn1_rass, model_rnn2_rass])
    model_rass.eval()
    if use_gpu:
        model_rass = model_rass.cuda()
    else:
        model_rass = model_rass.cpu()
    
    ## feed data into RASS model
        
    ypz, rnn2_last_layer, _, rnn1_yp, rnn1_last_layer, _, cnn_yp, cnn_last_layer = model_rass(X, return_last=False, return_ordinal_z=True)
    ypz = var_to_np(ypz).astype(float)
    ypz = ypz[...,0]
    Nrec = len(ypz)
    ordinal_thres = model_rass.modules[-1].output_layer.get_mus().astype(float)
    yp2d = model_rass.modules[-1].output_layer.get_proba(ypz.flatten())
    yp2d = yp2d.reshape(Nrec, -1, yp2d.shape[-1])
    yp1d = np.argmax(yp2d, axis=2)-5  # -5 to convert to RASS scale
    
    ## save the RASS results
    
    # ypz:  the predicted z-score (continuous value before discretizing into RASS levels), shape=(#recording, #segment)
    # yp1d: the predicted RASS, shape=(#recording, #segment)
    # yp2d: the predicted probability of RASS, shape=(#recording, #segment, 6), the first column is RASS -5, ...the last columns is RASS 0.
    sio.savemat('RASS_output.mat',
            {'ypz':ypz, 'yp1d': yp1d, 'yp2d': yp2d,
            'ordinal_thres':ordinal_thres,
            'specs_db': specs_db, 'freqs':freqs})
    
    
    ## load the CAM-ICU model
    
    # works under Python 2.7 and PyTorch 0.4.0
    model_cnn_camicu = EEGNet_CNN_CAMICU(32)
    model_cnn_camicu.load_state_dict(th.load('models/model_CAMICU_cnn.pth'))
    model_rnn1_camicu = EEGNet_RNN('bin', 192, 2, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=8)
    model_rnn1_camicu.load_state_dict(th.load('models/model_CAMICU_rnn1.pth'))
    model_rnn2_camicu = EEGNet_RNN('bin', 8, 2, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=4)
    model_rnn2_camicu.load_state_dict(th.load('models/model_CAMICU_rnn2.pth'))
    model_camicu = ModuleSequence([model_cnn_camicu, model_rnn1_camicu, model_rnn2_camicu])
    model_camicu.eval()
    if use_gpu:
        model_camicu = model_camicu.cuda()
    else:
        model_camicu = model_camicu.cpu()
    
    ## feed data into CAM-ICU model
    
    yp1d, rnn2_last_layer, _, rnn1_yp, rnn1_last_layer, _, cnn_yp, cnn_last_layer = model_camicu(X, return_last=False)
    yp1d = var_to_np(yp1d).astype(float)
    yp1d = yp1d[...,0]
    
    ## save the CAM-ICU results
    
    # yp1d: the predicted probability of CAM-ICU being 1 (delirium), shape=(#recording, #segment)
    sio.savemat('CAMICU_output.mat', {'yp1d': yp1d})

