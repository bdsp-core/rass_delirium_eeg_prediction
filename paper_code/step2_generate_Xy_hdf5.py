from collections import Counter
import datetime
import os
import pickle
import h5py
import sys
import numpy as np
import scipy.io as sio
sys.path.insert(0, 'data')
from gather_patient_data import *

data_type = 'segs'
label_type = sys.argv[1]#'rass','camicu'
network_type = sys.argv[2]#'feedforward', 'recurrent'
use_labeled_only = True
input_segs_path = '/data/delirium/features_all'
dtypef = 'float16'
dtypei = 'int32'
Fs = 250.
window_time = 4  # [s]
step_time = 2  # [s]
recurrent_crop_time = 570.  # [s]
if network_type=='recurrent':
    output_h5_path = '/data/delirium/eeg_%s_%s_%s_w%gs%g_L%.1fmin.h5'%(data_type, label_type, network_type, window_time, step_time, recurrent_crop_time/60.)
elif network_type=='feedforward':
    output_h5_path = '/data/delirium/eeg_%s_%s_%s_w%gs%g.h5'%(data_type, label_type, network_type, window_time, step_time)


def get_recurrent_segs(lookback_time, eeg, yin, label_timesin, start_pos_in, drugsin, artifact_remark):
    lookback = int(round(lookback_time*Fs))
    window = int(round(window_time*Fs))
    step = int(round(step_time*Fs))
    lookback_windows = (lookback-window)//step
    L = 283
    max_artifact_ratio = 0.5
    bad_reasons = []
    take_every_window = 30  # 1min

    _, idx = np.unique(label_timesin, return_index=True)
    unique_label_times = label_timesin[np.sort(idx)]
    
    # first round, only record xstart
    xstarts = []
    for ut in unique_label_times:
        assess_ids = np.where(label_timesin==ut)[0]
        assert np.all(np.diff(assess_ids)==1)
        if np.sum(artifact_remark[assess_ids]!='normal')*1./len(assess_ids)>max_artifact_ratio:
            bad_reasons.append('Artifact more than %d%%'%(max_artifact_ratio*100.,))
            continue
        #possible_xstarts: those after lookback_time is still within this assess
        possible_xstarts = np.where(start_pos_in[assess_ids]+lookback<start_pos_in[assess_ids].max()+window)[0]
        if len(possible_xstarts)<=0:
            bad_reasons.append('label assess too short')
            continue
            
        for xstart in possible_xstarts[::take_every_window]:
            # xend is last one which is within start_pos_in[assess_ids][xstart]+lookback_time
            xend = np.where(start_pos_in[assess_ids]<start_pos_in[assess_ids][xstart]+step*lookback_windows)[0].max()+1
            if xend-xstart!=L:
                bad_reasons.append('not L')
                continue
            yy = yin[assess_ids][xstart:xend]
            notallnancols = np.where(~np.all(np.isnan(yy), axis=0))[0]
            if len(notallnancols)>0 and np.any(np.nanmax(yy[:,notallnancols], axis=0)!=np.nanmin(yy[:,notallnancols], axis=0)):
                bad_reasons.append('multiple ys')
                continue
            lt = label_timesin[assess_ids][xstart:xend]
            if len(np.unique(lt))>1:
                bad_reasons.append('multiple label_times')
                continue
            xstarts.append(assess_ids[xstart])
    xstarts = np.array(xstarts)

    if len(bad_reasons)>0:
        print(Counter(bad_reasons))

    if len(xstarts)<=1:
        return [],[],[],[],[]
    
    # second round, get the data
    #try:
    #    X = eeg[list(map(lambda x:np.arange(x,x+L),xstarts))][:,::2]
    #except Exception as eee:
    X = eeg[list(map(lambda x:np.arange(x,x+L),xstarts)),:][:,::2]
    start_pos_out = (start_pos_in.reshape(-1,1)[list(map(lambda x:np.arange(x,x+L),xstarts)),:]).squeeze()[:,::2]
    
    yout = yin[xstarts]
    label_timesout = label_timesin[xstarts]
    
    #start_pos_in_list = start_pos_in.tolist()
    start_pos2 = (np.arange(len(start_pos_in)).reshape(-1,1)[list(map(lambda x:np.arange(x,x+283),xstarts)),:]).squeeze()[:,::2]#np.vectorize(lambda x:start_pos_in_list.index(x) if x>=0 else -1)(start_pos_out)
    #drugsout = np.array([drugsin[start_pos2[ii,:]] for ii in range(len(X))])
    drugsout = drugsin[start_pos2]
    drugsout[np.isnan(drugsout)] = 0.
    
    return X, yout, label_timesout, start_pos_out, drugsout
    

def get_feedforward_segs(eeg, yin, label_timesin, start_pos_in, drugsin, artifact_remark):
    X = []
    yout = []
    label_timesout = []
    start_pos_out = []
    #bad_reasons = []
    
    # remove overlapping in EEG when window=4s, step=2s to reduce data size and fit into memory...
    # do this first, otherwise if to remove artifact, cannot keep 2 consecutive samples are consecutive in time
    #eeg = eeg[::2]
    #yin = yin[::2]
    #label_timesin = label_timesin[::2]
    #start_pos_in = start_pos_in[::2]
    #drugsin = drugsin[::2]
    #artifact_remark = artifact_remark[::2]
    
    # remove artifact when supervised
    if label_type!='all':
        good_ids = np.where(artifact_remark=='normal')[0]
        X = eeg[good_ids]
        yout = yin[good_ids]
        label_timesout = label_timesin[good_ids]
        start_pos_out = start_pos_in[good_ids]
    # do not remove artifact when unsupervised
    else:
        X = eeg
        yout = yin
        label_timesout = label_timesin
        start_pos_out = start_pos_in
    if len(X)<=1:
        return [],[],[],[],[]
    
    start_pos_in_list = start_pos_in.tolist()
    start_pos2 = np.vectorize(lambda x:start_pos_in_list.index(x) if x>=0 else -1)(start_pos_out)
    drugsout = drugsin[start_pos2]
    drugsout[np.isnan(drugsout)] = 0.
    
    return X, yout, label_timesout, start_pos_out, drugsout
    
    
if __name__=='__main__':
    chunk_size = 64  # np.random.rand(512,1000,4).astype(dtypef).nbytes ~~ 1MB = 1x1024x1024x8 bytes
    patients = filter(lambda x:x.startswith('icused'), os.listdir(input_segs_path))
    patients = sorted(patients, key=lambda x:int(x[6:]))
    
    RecordID2PatientID = generate_RecordID2PatientID()
    MRN2PatientID = generate_MRN2PatientID(RecordID2PatientID=RecordID2PatientID)
    
    #or_data = prepare_OR_data(MRN2PatientID=MRN2PatientID)
    demographics_data = prepare_demographics_data(MRN2PatientID=MRN2PatientID, RecordID2PatientID=RecordID2PatientID, save_path='data/demographics.csv')
    weight_data, height_data = prepare_height_weight_data(RecordID2PatientID=RecordID2PatientID, weight_save_path='data/weights.csv', height_save_path='data/heights.csv')
    drug_data = prepare_drug_data(MRN2PatientID=MRN2PatientID, demographics_data=demographics_data, exclude_drug=['Norepinephrine','Phenylephrine'], save_path='data/drugs.csv')
    all_drugs = drug_data.Drug.unique().astype(str)
    #event_data = prepare_event_data(RecordID2PatientID=RecordID2PatientID, save_path='events.csv')
    
    artifact_counter = Counter([])
    with h5py.File(output_h5_path, 'w') as f:
        seg_num = 0
        for pid, patient in enumerate(patients):
            #try:
            patient_times, patient_label, patient_label_info, patient_eeg, patient_spect, patient_bsr, patient_artifact_remark, patient_drug, patient_demo = gather_patient_data(patient, demographics_data, drug_data, weight_data, height_data, with_eeg=True, with_spect=False, with_bsr=False, nan_pad=0 if use_labeled_only else None)
            patient_eeg = patient_eeg[:,:2]  # take 'Fp1-F7','Fp2-F8' to fit memeory
            
            #except Exception as ee:
            #    print(ee.message)
            #    continue
            patient_drug = np.array([patient_drug.get(dd, np.zeros(len(patient_times))+np.nan) for dd in all_drugs]).T
            if len(patient_times)<=0:
                continue
            
            if use_labeled_only:
                if label_type=='rass':
                    keep_ids = np.where(~np.isnan(patient_label[:,0]))[0]
                elif label_type=='camicu':
                    keep_ids = np.where(~np.isnan(patient_label[:,-1]))[0]
            else:
                keep_ids = np.arange(len(patient_times))
           
            patient_label_info = patient_label_info[keep_ids][::2]
            
            if use_labeled_only:
                # for each label type, only take its own label info
                # for example, 'rass...camicu...' --> 'rass...' for rass
                # this removes discontinuous such as rass1, rass1, rass1camicu1, rass1camicu1, rass1
                patient_label_info2 = []
                # number of elements for each label type, for example
                # rass\t1900-01-01 19:00:00.0\t0\tgood --> 4 elements
                # camicu\t1900-01-01 19:00:00.0\tLauren --> 3 elements
                label_type_n_elements = {'rass':4, 'camicu':3}
                for li, pli in enumerate(patient_label_info):
                    #if pli=='':
                    #    patient_label_info2.append('')
                    #else:
                    label_elements = map(lambda x: x.lower(), map(lambda x: x.strip(), pli.split('\t')))
                    #if label_type in label_elements:
                    label_type_id = label_elements.index(label_type)
                    label_elements = pli.split('\t')
                    patient_label_info2.append('\t'.join([label_elements[0]]+label_elements[label_type_id:label_type_id+label_type_n_elements[label_type]]))
                    #else:
                    #    patient_label_info2.append('')
                patient_label_info = np.array(patient_label_info2)        
            
            patient_times = patient_times[keep_ids][::2]  # TODO only when window_step = 2s
            patient_label = patient_label[keep_ids][::2]
            if patient_eeg is not None:
                patient_eeg = patient_eeg[keep_ids][::2]
            if patient_spect is not None:
                patient_spect = patient_spect[keep_ids][::2]
            if patient_bsr is not None:
                patient_bsr = patient_bsr[keep_ids][::2]
            patient_artifact_remark = patient_artifact_remark[keep_ids][::2]
            artifact_counter += Counter(patient_artifact_remark)
            patient_drug = patient_drug[keep_ids][::2]
            
            if len(patient_times)<=1:
                continue
            seg_start_pos_in = ((patient_times-patient_times.min())*Fs).astype(int)
            
            if not use_labeled_only:
                sio.savemat('/data/delirium/%s.mat'%patient,
                    {'EEG':np.ascontiguousarray(patient_eeg[::2]), 'label':np.ascontiguousarray(patient_label[::2]),
                    'label_info':np.ascontiguousarray(patient_label_info[::2]),
                    'artifact':np.ascontiguousarray(patient_artifact_remark[::2])})
                import pdb;pdb.set_trace()
                continue

            
            if network_type=='feedforward':
                # X.shape = N x 4 x 1000
                X, y, label_times, seg_start_pos, drugs = get_feedforward_segs(patient_eeg, patient_label, patient_label_info, seg_start_pos_in, patient_drug, patient_artifact_remark)
            elif network_type=='recurrent':
                # X.shape = N x T x 4 x 1000
                X, y, label_times, seg_start_pos, drugs = get_recurrent_segs(recurrent_crop_time, patient_eeg, patient_label, patient_label_info, seg_start_pos_in, patient_drug, patient_artifact_remark)#consecutive_example_ids, n_lookback
            else:
                raise ValueError(network_type)
        
            if len(X)<=1:
                continue
                    
            label_times = label_times.astype(str)
            #assessments = np.array(map(lambda x:'%s_%d'%(record_name,x), assessments))

            print('[%d/%d %s]\t%s\t%d\t%d'%(pid+1, len(patients), datetime.datetime.now(), patient, len(X), len(np.unique(label_times))))

            if 'X' not in f:  # first write
                #dtypes1 = np.array([record_name+'                           ']).dtype  # to be enough to put long record names
                dtypes2 = np.array([patient+'   ']).dtype
                dtypes3 = np.array([' '*200]).dtype
                
                seg_num = X.shape[0]
                dX = f.create_dataset('X', shape=X.shape, maxshape=(None,)+X.shape[1:],
                                        chunks=(chunk_size,)+X.shape[1:], dtype=dtypef)
                dy = f.create_dataset('y', shape=y.shape, maxshape=(None,)+y.shape[1:],
                                        chunks=True, dtype=dtypef)
                ddrug = f.create_dataset('drug', shape=drugs.shape, maxshape=(None,)+drugs.shape[1:],
                                        chunks=(chunk_size,)+drugs.shape[1:], dtype=dtypef)
                ddrugname = f.create_dataset('drugname', shape=all_drugs.shape, dtype=all_drugs.dtype)
                dlabel_times = f.create_dataset('label_times', shape=label_times.shape, maxshape=(None,),
                                        chunks=True, dtype=dtypes3)
                dstartpos = f.create_dataset('seg_start_pos', shape=seg_start_pos.shape, maxshape=(None,)+seg_start_pos.shape[1:],
                                        chunks=True, dtype=dtypei)
                #drecords = f.create_dataset('record', shape=(X.shape[0],), maxshape=(None,),
                #                        chunks=True, dtype=dtypes1)
                #dassessments = f.create_dataset('assessment', shape=assessments.shape, maxshape=(None,)+assessments.shape[1:],
                #                        chunks=True, dtype=dtypes1)
                dpatients = f.create_dataset('patient', shape=(X.shape[0],), maxshape=(None,),
                                        chunks=True, dtype=dtypes2)
                #if network_type=='recurrent':
                #    dlengths = f.create_dataset('lengths', shape=lengths.shape, maxshape=(None,)+lengths.shape[1:],
                #                        chunks=True, dtype=dtypei)
                if data_type=='specs':
                    dfreq = f.create_dataset('EEG_freq', shape=freq.shape,# fixed size
                                        dtype=dtypef)
                                        
                dX[:] = X
                dy[:] = y
                dlabel_times[:] = label_times
                ddrug[:] = drugs
                ddrugname[:] = all_drugs
                dstartpos[:] = seg_start_pos
                #dassessments[:] = assessments
                #drecords[:] = [record_name]*X.shape[0]
                dpatients[:] = [patient]*X.shape[0]
                #if network_type=='recurrent':
                #    dlengths[:] = lengths
                if data_type=='specs':
                    dfreq[:] = freq
            else:
                dX.resize(seg_num + X.shape[0], axis=0)
                dy.resize(seg_num + X.shape[0], axis=0)
                ddrug.resize(seg_num + X.shape[0], axis=0)
                dlabel_times.resize(seg_num + X.shape[0], axis=0)
                #dassessments.resize(seg_num + X.shape[0], axis=0)
                dstartpos.resize(seg_num + X.shape[0], axis=0)
                #drecords.resize(seg_num + X.shape[0], axis=0)
                dpatients.resize(seg_num + X.shape[0], axis=0)
                #if network_type=='recurrent':
                #    dlengths.resize(seg_num + X.shape[0], axis=0)
                
                dX[seg_num:] = X
                dy[seg_num:] = y
                ddrug[seg_num:] = drugs
                dlabel_times[seg_num:] = label_times
                dstartpos[seg_num:] = seg_start_pos
                #dassessments[seg_num:] = assessments
                #drecords[seg_num:] = [record_name]*X.shape[0]
                dpatients[seg_num:] = [patient]*X.shape[0]
                #if network_type=='recurrent':
                #    dlengths[seg_num:] = lengths

                seg_num += X.shape[0]
        
        # print artifact ratios
        with open('artifact_counter_%s_%s.pickle'%(data_type, network_type), 'wb') as ff:
            pickle.dump(artifact_counter, ff, protocol=2)
        for ac in artifact_counter:
            print('%s: %d/%d, %.2f%%'%(ac, artifact_counter[ac],
                    sum(artifact_counter.values()),
                    artifact_counter[ac]*100./sum(artifact_counter.values())))
                   
