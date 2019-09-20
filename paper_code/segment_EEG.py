#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import detrend
import mne
mne.set_log_level(verbose='WARNING')
from mne.filter import filter_data, notch_filter
from peakdetect import peakdetect
from read_delirium_data import datenum


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


def segment_EEG(EEG_, rass_labels_, camicu_labels_, assess_time_before, assess_time_after,
                times, t0s, t1s, window_time, step_time,
                Fs, notch_freq=None, bandpass_freq=None, start_end_remove_window_num=0,
                amplitude_thres=500, n_jobs=1, to_remove_mean=False):#
    """
    Segment EEG signals.
    """
    std_thres1 = 0.2
    std_thres2 = 0.5
    flat_seconds = 2
    
    EEG = EEG_[[0,1,3,4]]  # not use the referential electrode
    
    if to_remove_mean:
        EEG = EEG - np.mean(EEG,axis=1, keepdims=True)
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    flat_length = int(round(flat_seconds*Fs))
    assess_size_before = int(round(assess_time_before*Fs))
    assess_size_after = int(round(assess_time_after*Fs))
                
    ## generate labels
    
    assess_times = np.array(['']*EEG.shape[1], dtype=object)
    labels = np.zeros((EEG.shape[1],6))+np.nan
    
    # RASS label: first remove assess <1h apart but with different scores
    rass_times_sec = np.array(list(map(lambda xx:datenum(xx, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True), rass_labels_[:,2])))
    bad_ids = np.where((rass_times_sec[1:]-rass_times_sec[:-1]<3600)&(rass_labels_[1:,3]!=rass_labels_[:-1,3]))[0]
    bad_ids = np.sort(np.r_[bad_ids, bad_ids+1])
    good_ids = np.setdiff1d(range(len(rass_labels_)), bad_ids)
    rass_labels_ = rass_labels_[good_ids]
    rass_times_sec = rass_times_sec[good_ids]

    # RASS label: then generate labels by [loc - assess_before, loc + assess_before]
    for i in range(len(rass_labels_)):
        if '00:00:00' in rass_labels_[i,2] and rass_labels_[i,2] in camicu_labels_[:,3]:# rass_labels_[i,5]!='good' or rass_labels_[i,4]!=0
            continue
        loc = int(round((rass_times_sec[i]-times[0])*Fs))
        start = max(0, loc-assess_size_before)
        end = min(EEG.shape[1], loc+assess_size_after)
        if end-start<=0:
            continue
        # exist overlap if np.any(~np.isnan(labels[start:end,0])):
        labels[start:end,0] = float(rass_labels_[i,3])
        assess_times[start:end] = '\t'.join(['RASS', rass_labels_[i,2], rass_labels_[i,4], rass_labels_[i,5]])
    
    # CAMICU label: first remove assess <1h apart but with different scores
    camicu_times_sec = np.array(list(map(lambda xx:datenum(xx, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True), camicu_labels_[:,1])))
    bad_ids = np.where((camicu_times_sec[1:]-camicu_times_sec[:-1]<3600)&(camicu_labels_[1:,9]!=camicu_labels_[:-1,9]))[0]
    bad_ids = np.sort(np.r_[bad_ids, bad_ids+1])
    good_ids = np.setdiff1d(range(len(camicu_labels_)), bad_ids)
    camicu_labels_ = camicu_labels_[good_ids]
    camicu_times_sec = camicu_times_sec[good_ids]
    
    # CAMICU label: then generate labels by [loc - assess_before, loc + assess_before]
    mystrip=np.vectorize(lambda x:x.strip())
    max_beyond = int(round(3*3600*Fs))  # max beyond 3h
    dist_thres = int(round(15*60*Fs))  # threshold is be considered as close/far
    for i in range(len(camicu_labels_)):
        if '00:00:00' in camicu_labels_[i,1]:
            continue
        # if the corresponding RASS is >0, it is hyperactive delirium, remove
        if camicu_labels_[i,1] in rass_labels_[:,2] and float(rass_labels_[list(rass_labels_[:,2]).index(camicu_labels_[i,1]),3])>0:
            continue
        loc = int(round((camicu_times_sec[i]-times[0])*Fs))
        
        # if this assessment is always outside recordings OR inside a short recording
        inside = (camicu_times_sec[i]>=t0s)&(camicu_times_sec[i]<=t1s)
        can_be_beyond = ~np.any(inside)
        if not can_be_beyond:
            inside_id = np.where(inside)[0][0]
            if t1s[inside_id]-t0s[inside_id]<10*60:
                can_be_beyond = True
        
        start = None
        end = None
        # if within [before, EEG.shape[1]-after]
        if assess_size_before<=loc<=EEG.shape[1]-assess_size_after:
            start = loc-assess_size_before
            end = loc+assess_size_after
        # if within [0, before]
        elif 0<=loc<assess_size_before:
            start = 0
            end = assess_size_before+assess_size_after
        # if within [EEG.shape[1]-after, EEG.shape[1]]
        elif EEG.shape[1]-assess_size_after<loc<EEG.shape[1]:
            start = EEG.shape[1]-assess_size_before-assess_size_after
            end = EEG.shape[1]
        # if within [-max_beyond, 0]
        elif -max_beyond<loc<0 and can_be_beyond:
            # decide the distance with the previous EEG recording
            dists = t0-t1s[patient_ids==patient_id]
            # recordings before this one have dists>=0
            dists_pos = dists[dists>=0]
            if len(dists_pos)==0:
                # this is the first recording, take the first part of recording
                start = 0
                end = assess_size_before+assess_size_after
            else:
                # the one right before this one has dist=min(dists_pos)
                dist = int(round(dists_pos.min()*Fs))
                if dist<=dist_thres:
                    # there is a previous recording close to this
                    start = 0
                    end = assess_size_after
                # there is a previous recording far to this
                # if loc is closer to this recording
                elif np.abs(loc)<dist/2.:
                    start = 0
                    end = assess_size_before+assess_size_after
        # if within [EEG.shape[1], EEG.shape[1]+max_beyond]
        elif EEG.shape[1]<=loc<EEG.shape[1]+max_beyond and can_be_beyond:
            # decide the distance with the next EEG recording
            dists = t0s[patient_ids==patient_id]-t1
            # recordings after this one have dists>=0
            dists_pos = dists[dists>=0]
            if len(dists_pos)==0:
                # this is the last recording, take the last part of recording
                start = EEG.shape[1]-assess_size_before-assess_size_after
                end = EEG.shape[1]
            else:
                # the one right after this one has dist=min(dists_pos)
                dist = int(round(dists_pos.min()*Fs))
                if dist<=dist_thres:
                    # there is a next recording close to this
                    start = EEG.shape[1]-assess_size_after
                    end = EEG.shape[1]
                # there is a previous recording far to this
                # if loc is closer to this recording
                elif np.abs(loc-EEG.shape[1])<dist/2.:
                    start = EEG.shape[1]-assess_size_before-assess_size_after
                    end = EEG.shape[1]
                    
        if start is None or end is None:
            continue
        # exist overlap if np.any(~np.isnan(labels[start:end,0])):
        labels[start:end,1:] = camicu_labels_[i,[3,4,7,8,9]].astype(float)
        assess_times[start:end] = mystrip(assess_times[start:end]+'\t'.join(['','CAMICU', camicu_labels_[i,1], str(camicu_labels_[i,2])]))
    
    ## start_ids
    
    start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    if len(start_ids) <= 0:
        raise ValueError('No EEG segments')
    
    seg_masks = [seg_mask_explanation[0]]*len(start_ids)
    
    ## apply montage (interchangeable with linear filtering)
        
    EEG = EEG[[0,1,0,2]] - EEG[[2,3,1,3]]
    
    ## filter signal
    
    EEG = notch_filter(EEG, Fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size)
    EEG = filter_data(EEG, Fs, bandpass_freq[0], bandpass_freq[1], n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size)
    
    ## detect burst suppression
    
    EEG_mne = mne.io.RawArray(np.array(EEG, copy=True), mne.create_info(EEG.shape[0], Fs, ch_types='eeg', verbose='ERROR'), verbose='ERROR')
    EEG_mne.apply_hilbert(envelope=True, n_jobs=-1, verbose='ERROR')
    BS = EEG_mne.get_data()
    
    bs_window_size = int(round(120*Fs))  # BSR estimation window 2min
    bs_start_ids = np.arange(0, EEG.shape[1]-bs_window_size+1,bs_window_size)
    if len(bs_start_ids)<=0:
        bs_start_ids = np.array([0], dtype=int)
    if EEG.shape[1]>bs_start_ids[-1]+bs_window_size:  # if incomplete divide
        bs_start_ids = np.r_[bs_start_ids, EEG.shape[1]-bs_window_size]
    BS_segs = BS[:,map(lambda x:np.arange(x,min(BS.shape[1],x+bs_window_size)), bs_start_ids)]
    BSR_segs = np.sum(BS_segs<=5, axis=2).T*1./bs_window_size
    BSR = np.zeros_like(EEG)
    for ii, bsi in enumerate(bs_start_ids):
        BSR[:, bsi:min(BSR.shape[1], bsi+bs_window_size)] = BSR_segs[ii].reshape(-1,1)
    
    ## segment signal

    BSR_segs = BSR[:,map(lambda x:np.arange(x,x+window_size), start_ids)].transpose(1,0,2).mean(axis=2)
    EEG_segs = EEG[:,map(lambda x:np.arange(x,x+window_size), start_ids)].transpose(1,0,2)  # (#window, #ch, window_size)
    
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
    specs = 10*np.log10(specs.transpose(0,2,1))
    
            
    return EEG_segs, BSR_segs, labels[start_ids], np.array(assess_times)[start_ids].tolist(), start_ids, seg_masks, specs, freq

