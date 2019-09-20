#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
import datetime
import os
import h5py
import hdf5storage as hs
import scipy.io as sio
import numpy as np
import pandas as pd
from read_delirium_data import *
from segment_EEG import *


Fs = 250.  # [Hz]
assess_time_before = 1800  # [s]
assess_time_after = 1800  # [s]
window_time = 4  # [s]
window_step = 2  # [s]
start_end_remove_window_num = 1  # remove windows at start and end
amplitude_thres = 500  # [uV]
line_freq = 60.  # [Hz]
bandpass_freq = [0.5, 20.]  # [Hz]
available_channels = ['FP1', 'FP2', 'FPZ', 'F7', 'F8']
eeg_channels = ['Fp1-F7','Fp2-F8','Fp1-Fp2','F7-F8']
random_state = 1
normal_only = False

seg_mask_explanation = np.array([
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
    '1Hz artifact',])


if __name__=='__main__':
    ##################
    # use data_list_paths to specify what data to use
    # note: eeg segments are contained in features column
    ##################
    
    data_list_path = 'data/data_list.txt'
    subject_files = pd.read_csv(data_list_path, sep='\t')
    subject_files = subject_files[subject_files.status=='good'].reset_index(drop=True)
    patient_ids = np.array([[x for x in xx.split(os.path.sep) if x.startswith('icused')][0] for xx in subject_files.eeg])
    t0s = np.array([datenum(t0str, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True) for t0str in subject_files.t0])
    t1s = np.array([datenum(t1str, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True) for t1str in subject_files.t1])
    record_num = subject_files.shape[0]
    """
    # get the recording interval distribution
    dists = []
    for pid in np.unique(patient_ids):
        tt0 = t0s[patient_ids==pid]
        tt1 = t1s[patient_ids==pid]
        ids = np.argsort(tt0)
        tt0 = tt0[ids]
        tt1 = tt1[ids]
        assert np.all(np.diff(tt1)>0)
        assert np.all(tt1-tt0>0)
        dists.extend(tt0[1:]-tt1[:-1])
    plt.hist(np.log1p(dists),bins=50);plt.show()
    """
    
    # load subjects with error if exists
    subject_err_path = 'data/err_subject_reason.txt'
    if os.path.isfile(subject_err_path):
        err_subject_reason = []
        with open(subject_err_path,'r') as f:
            for row in f:
                if row.strip()=='':
                    continue
                i = row.split(':::')
                err_subject_reason.append([i[0].strip(), i[1].strip()])
        err_subject = [i[0] for i in err_subject_reason]
    else:
        err_subject_reason = []
        err_subject = []

    all_rass_times = np.loadtxt('data/rass_times.txt', dtype=str, delimiter='\t', skiprows=1)
    all_camicu_times = pd.read_csv('data/vICU_Sed_CamICU.csv', sep=',')
    # loop over all subjects
    for si in range(record_num):
    
        data_path = subject_files.eeg[si]
        feature_path = subject_files.feature[si]
        t0 = t0s[si]
        t1 = t1s[si]
        patient_id = patient_ids[si]
        subject_file_name = os.path.join(patient_id, data_path.split(os.path.sep)[-1])
        if subject_file_name in err_subject:
            continue
            
        # if exists, skip
        if os.path.isfile(feature_path):
            print('\n[(%d)/%d] %s %s'%(si+1,record_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))

        # other generate the output file
        else:
            print('\n[%d/%d] %s %s'%(si+1,record_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))
            try:
                # load dataset
                res = read_delirium_mat(data_path, channel_names=available_channels)
                if res['Fs']<Fs-1 or res['Fs']>Fs+1:
                    raise ValueError('Fs is not %gHz.'%Fs)

                # segment EEG
                segs_, bs_, labels_, assessment_times_, seg_start_ids_, seg_mask, specs_, freqs_ = segment_EEG(res['data'],
                        all_rass_times[all_rass_times[:,1]==patient_id,:],
                        all_camicu_times[all_camicu_times.PatientID==patient_id].values,
                        assess_time_before, assess_time_after,
                        [t0,t1], t0s[patient_ids==patient_id], t1s[patient_ids==patient_id], window_time, window_step, Fs,
                        notch_freq=line_freq, bandpass_freq=bandpass_freq,
                        to_remove_mean=False, amplitude_thres=amplitude_thres, n_jobs=-1, start_end_remove_window_num=start_end_remove_window_num)
                if len(segs_) <= 0:
                    raise ValueError('No segments')
                
                # print info
                print('\n%s\n'%Counter(labels_[np.logical_not(np.isnan(labels_))]))
                seg_mask2 = map(lambda x:x.split('_')[0], seg_mask)
                sm = Counter(seg_mask2)
                for ex in seg_mask_explanation:
                    if ex in sm:
                        print('%s: %d/%d, %g%%'%(ex,sm[ex],len(seg_mask),sm[ex]*100./len(seg_mask)))
                
                if normal_only:
                    good_ids = np.where(np.array(seg_mask)=='normal')[0]
                    segs_ = segs_[good_ids]
                    bs_ = bs_[good_ids]
                    labels_ = labels_[good_ids]
                    assessment_times_ = [assessment_times_[ii] for ii in good_ids]
                    seg_start_ids_ = seg_start_ids_[good_ids]
                    seg_mask = [seg_mask[ii] for ii in good_ids]
                    specs_ = specs_[good_ids]
                if segs_.shape[0]<=0:
                    raise ValueError('No EEG signal')
                if segs_.shape[1]!=len(eeg_channels):
                    raise ValueError('Incorrect #chanels')

            except Exception as e:
                # if any error, append to subjects with error
                err_info = e.message.split('\n')[0].strip()
                print('\n%s.\nSubject %s is IGNORED.\n'%(err_info,subject_file_name))
                err_subject_reason.append([subject_file_name,err_info])
                err_subject.append(subject_file_name)

                with open(subject_err_path,'a') as f:
                    msg_ = '%s::: %s\n'%(subject_file_name,err_info)
                    f.write(msg_)
                continue

            # save into output feature file
            fd = os.path.split(feature_path)[0]
            if not os.path.exists(fd):
                os.mkdir(fd)
            res = {'EEG_segs':segs_.astype('float32'),
                'EEG_specs':specs_.astype('float32'),
                'burst_suppression':bs_.astype('float32'),
                'EEG_frequency':freqs_,
                't0':t0str,
                't1':t1str,
                'labels':labels_,
                'assess_times':assessment_times_,
                'seg_start_ids':seg_start_ids_,
                'subject':subject_file_name,
                'Fs':Fs}
            if not normal_only:
                res['seg_masks'] = seg_mask

            # looks like it's faster to save the file in this way
            sio.savemat(feature_path, res, do_compression=True)
            res = sio.loadmat(feature_path)
            os.remove(feature_path)
            hs.savemat(feature_path, res)
                

