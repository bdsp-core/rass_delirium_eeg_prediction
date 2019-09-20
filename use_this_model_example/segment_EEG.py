#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import detrend
from scikits.samplerate import resample
import mne
mne.set_log_level(verbose='WARNING')
from mne.filter import filter_data, notch_filter
from peakdetect import peakdetect


seg_mask_explanation = [
    'normal',
    'NaN in EEG', #_[1,3] (append channel ids)
    'overly high/low amplitude',
    'flat signal',
    'NaN in feature',
    'NaN in spectrum',
    'overly high/low total power',
    'muscle artifact',
    'multiple assessment scores',
    'spurious spectrum',
    'fast rising decreasing',
    '1Hz artifact',]
    
    
def peak_detect_num_amp(signal, lookahead=200, delta=0):
    # signal: #channel x #points
    res_num = []
    res_amp = []
    for cid in range(signal.shape[0]):
        local_max, local_min = peakdetect(signal[cid], lookahead=lookahead, delta=delta)
        if len(local_min)<=0:
            local_extremes = np.array(local_max)
        elif len(local_max)<=0:
            local_extremes = np.array(local_min)
        else:
            local_extremes = np.r_[local_max, local_min]
        res_num.append(len(local_extremes))
        if len(local_extremes)<=0:
            amp = 0
        else:
            amp = local_extremes[:,1].max()-local_extremes[:,1].min()
        res_amp.append(amp)
    return res_num, res_amp
    
    
def peak_detect(signal, max_change_points, min_change_amp, lookahead=200, delta=0):
    # signal: #channel x #points
    res = []
    for cid in range(signal.shape[0]):
        local_max, local_min = peakdetect(signal[cid], lookahead=lookahead, delta=delta)
        if len(local_min)<=0 and len(local_max)<=0:
            res.append(False)
        else:
            if len(local_min)<=0:
                local_extremes = np.array(local_max)
            elif len(local_max)<=0:
                local_extremes = np.array(local_min)
            else:
                local_extremes = np.r_[local_max, local_min]
            local_extremes = local_extremes[np.argsort(local_extremes[:,0])]
            res.append(np.logical_and(np.diff(local_extremes[:,0])<=max_change_points, np.abs(np.diff(local_extremes[:,1]))>=min_change_amp).sum())
    return res
    
    
def autocorrelate_noncentral_max_abs(x):
    ress = []
    for ii in range(x.shape[1]):
        res = np.correlate(x[:,ii],x[:,ii],mode='full')/np.correlate(x[:,ii],x[:,ii],mode='valid')[0]
        ress.append(np.max(res[len(res)//2+7:len(res)//2+20]))  # ECG range: 40/1min(0.7Hz) -- 120/1min(2Hz)
    return ress

    segs_, seg_start_ids_, seg_mask, specs_, freqs_ = segment_EEG(res['data'],
            window_time, window_step, res['Fs'], newFs,
            notch_freq=line_freq, bandpass_freq=bandpass_freq,
            amplitude_thres=amplitude_thres, n_jobs=-1)

def segment_EEG(EEG, window_time, step_time,
                Fs, newFs, notch_freq=None, bandpass_freq=None,
                start_end_remove_window_num=0, amplitude_thres=500,
                n_jobs=1, to_remove_mean=False):
    """
    Segment EEG signals.
    """
    std_thres1 = 0.2
    std_thres2 = 0.5
    flat_seconds = 2
    
    # resample
    if newFs!=Fs:
        r = newFs*1./Fs
        EEG = Parallel(n_jobs=n_jobs, verbose=False)(delayed(resample)(EEG[i], r, 'sinc_best') for i in range(len(EEG)))
        EEG = np.array(EEG).astype(float)
        Fs = newFs
    
    if to_remove_mean:
        EEG = EEG - np.mean(EEG,axis=1, keepdims=True)
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    flat_length = int(round(flat_seconds*Fs))
    
    ## start_ids
    
    start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    if len(start_ids) <= 0:
        raise ValueError('No EEG segments')
    
    seg_masks = [seg_mask_explanation[0]]*len(start_ids)
    
    ## apply montage (interchangeable with linear filtering)
    EEG = EEG[[0,1]] - EEG[[2,3]]
    
    ## filter signal
    
    EEG = notch_filter(EEG, Fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size)
    EEG = filter_data(EEG, Fs, bandpass_freq[0], bandpass_freq[1], n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size)
    
    ## segment signal

    EEG_segs = EEG[:,list(map(lambda x:np.arange(x,x+window_size), start_ids))].transpose(1,0,2)  # (#window, #ch, window_size)
    
    ## find nan in signal
    
    nan2d = np.any(np.isnan(EEG_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        seg_masks[i] = '%s_%s'%(seg_mask_explanation[1], np.where(nan2d[i])[0])
        
    ## calculate spectrogram
    
    BW = 2.
    specs, freq = mne.time_frequency.psd_array_multitaper(EEG_segs, Fs, fmin=bandpass_freq[0], fmax=bandpass_freq[1], adaptive=False, low_bias=False, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
    df = freq[1]-freq[0]
    specs = 10*np.log10(specs.transpose(0,2,1))
    
    ## find nan in spectrum
    
    specs[np.isinf(specs)] = np.nan
    nan2d = np.any(np.isnan(specs), axis=1)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    nonan_spec_id = np.where(np.all(np.logical_not(np.isnan(specs)), axis=(1,2)))[0]
    for i in nan1d:
        seg_masks[i] = '%s_%s'%(seg_mask_explanation[5],np.where(nan2d[i])[0])
        
    ## find staircase-like spectrum
    # | \      +-+
    # |  \     | |
    # |   -----+ +--\
    # +--------------=====
    spec_smooth_window = int(round(1./df))  # 1 Hz
    specs2 = specs[nonan_spec_id][:,np.logical_and(freq>=5,freq<=20)]
    freq2 = freq[np.logical_and(freq>=5,freq<=20)][spec_smooth_window:-spec_smooth_window]
    ww = np.hanning(spec_smooth_window*2+1)
    ww = ww/ww.sum()
    smooth_specs = np.apply_along_axis(lambda m: np.convolve(m, ww, mode='valid'), axis=1, arr=specs2)
    dspecs = specs2[:,spec_smooth_window:-spec_smooth_window]-smooth_specs
    dspecs = dspecs-dspecs.mean(axis=1,keepdims=True)#()/dspecs_std
    aa = np.apply_along_axis(lambda m: np.convolve(m, np.array([-1.,-1.,0,1.,1.,1.,1.]), mode='same'), axis=1, arr=dspecs)  # increasing staircase-like pattern
    bb = np.apply_along_axis(lambda m: np.convolve(m, np.array([1.,1.,1.,1.,0.,-1.,-1.]), mode='same'), axis=1, arr=dspecs)  # decreasing staircase-like pattern
    stsp2d = np.logical_or(np.maximum(aa,bb).max(axis=1)>=10, np.any(np.abs(np.diff(specs2,axis=1))>=11, axis=1))
    stsp1d = nonan_spec_id[np.any(stsp2d, axis=1)]
    for i in stsp1d:
        seg_masks[i] = seg_mask_explanation[9]
    
    ## check ECG in spectrum (~1Hz and harmonics)

    autocorrelation = Parallel(n_jobs=n_jobs,verbose=True)(delayed(autocorrelate_noncentral_max_abs)(spec) for spec in dspecs)
    autocorrelation = np.array(autocorrelation)
    ecg2d = autocorrelation>0.7
    ecg1d = nonan_spec_id[np.any(ecg2d,axis=1)]
    for i in ecg1d:
        seg_masks[i] = seg_mask_explanation[11]
    
    ## find overly fast rising/decreasing signal
    
    max_change_points = 0.1*Fs
    min_change_amp = 1.8*amplitude_thres
    fast_rising2d = Parallel(n_jobs=n_jobs,verbose=True)(delayed(peak_detect)(EEG_segs[sid], max_change_points, min_change_amp, lookahead=50, delta=0) for sid in range(EEG_segs.shape[0]))
    fast_rising2d = np.array(fast_rising2d)>0
    fast_rising1d = np.where(np.any(fast_rising2d, axis=1))[0]
    for i in fast_rising1d:
        seg_masks[i] = seg_mask_explanation[10]
    
    ## find large amplitude in signal
            
    amplitude_large2d = np.max(EEG_segs,axis=2)-np.min(EEG_segs,axis=2)>2*amplitude_thres
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    for i in amplitude_large1d:
        seg_masks[i] = '%s_%s'%(seg_mask_explanation[2], np.where(amplitude_large2d[i])[0])
            
    ## find flat signal
    # careful about burst suppression
    
    short_segs = EEG_segs.reshape(EEG_segs.shape[0], EEG_segs.shape[1], EEG_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres1, axis=2)
    flat2d = np.logical_or(flat2d, np.std(EEG_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        seg_masks[i] = '%s_%s'%(seg_mask_explanation[3], np.where(flat2d[i])[0])
    
    BW = 1.  #frequency resolution 1Hz
    specs, freq = mne.time_frequency.psd_array_multitaper(EEG_segs, Fs, fmin=bandpass_freq[0], fmax=bandpass_freq[1], adaptive=False, low_bias=False, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
    specs_db = 10*np.log10(specs.transpose(0,2,1))
    
            
    return EEG_segs, start_ids, seg_masks, specs_db, freq

