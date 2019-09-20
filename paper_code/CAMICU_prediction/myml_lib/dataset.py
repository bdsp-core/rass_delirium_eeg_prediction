import copy
import sys
from collections import Counter
import datetime# import timedelta, strptime #datetime
import os.path
import numpy as np
import pandas as pd
from scipy.signal import detrend
from tqdm import tqdm
import h5py
import mne
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import Dataset
#from read_delirium_data import datenum

SECONDS_IN_DAY = 86400.


def datenum(date_str, format_, return_seconds=False):
    days = (datetime.datetime.strptime(date_str, format_)-datetime.datetime(1,1,1,0,0,0)).total_seconds()*1./SECONDS_IN_DAY+367.
    if return_seconds:
        return days*SECONDS_IN_DAY
    else:
        return days


class MyDataset(Dataset):
    def __init__(self, input_path, label_type, data_type='specs', inmemory=True, class_weighted=False, fliplr_prob=False, label_mapping=None, really_load=True):#, variable_names=None
        super(MyDataset, self).__init__()
        self.input_path = input_path
        self.label_type = label_type
        self.data_type = data_type
        self.inmemory = inmemory
        #self.pin_memory = pin_memory
        self.fliplr_prob = fliplr_prob
        self.class_weighted = class_weighted
        self.label_mapping = label_mapping
        self.len = 0
        self.repeated_times = 1
        self.noise_std = None
        self.shorten_amount = None
        self.shorten_padding = 0
        
        # load into memory
        with h5py.File(self.input_path, 'r') as data_source:
                
            if self.data_type=='specs':
                self.h5_len, self.time_step, self.D = data_source['X'].shape
                self.channel_num = 2####
                self.Dch = self.D//self.channel_num
            elif self.data_type=='eeg':
                if data_source['X'].ndim==3:
                    self.h5_len, self.channel_num, self.Dch = data_source['X'].shape
                else:
                    self.h5_len, self.time_step, self.channel_num, self.Dch = data_source['X'].shape
                self.D = self.Dch
                
            if not really_load:
                return

            self.y = data_source['y'][:]
            if self.label_type.lower()=='rass':
                self.select_ids = np.where(~np.isnan(self.y[:,0]))[0]
                self.y = self.y[self.select_ids, 0]
                if self.label_mapping is not None:
                    self.inv_label_mapping = {self.label_mapping[kk]:kk for kk in self.label_mapping}
                    tostudy_labels = self.label_mapping.keys()
                    tostudy_ids = np.where(np.in1d(self.y, tostudy_labels))[0]
                    self.y = np.array(map(lambda x:self.label_mapping[x], self.y[tostudy_ids]))
                    self.select_ids = self.select_ids[tostudy_ids]
                    
            elif self.label_type.lower()=='camicu':
                self.select_ids = np.where(~np.isnan(self.y[:,-1]))[0]
                self.y = self.y[self.select_ids]#, 1:]

            load_all_first = len(self.select_ids)>self.h5_len//10
            #if self.label_type.lower()=='rass':
            self.X = data_source['X'][:]
            #elif self.label_type.lower()=='camicu':
            #    if load_all_first:
            #        self.X = data_source['X'][:]
            #        self.X = self.X[self.select_ids]
            #    else:
            #        self.X = data_source['X'][list(self.select_ids)]
                    
            #if self.data_type=='eeg':
            #    if data_source['X'].ndim==3:
            #        self.X = self.X[:,:,::2]
            #    else:
            #        self.X = self.X[:,:,:,::2]
            #    self.D = self.D//2
            #    self.Dch = self.Dch//2
            
            if 'EEG_freq' in data_source:
                self.freqs = data_source['EEG_freq'][:]
                
            if 'patient' in data_source:
                if load_all_first:
                    self.patients = data_source['patient'][:]
                    if len(self.select_ids)<len(self.patients):
                        self.patients = self.patients[self.select_ids]
                else:
                    self.patients = data_source['patient'][list(self.select_ids)]
                
            if 'seg_start_pos' in data_source:
                if load_all_first:
                    self.seg_start_pos = data_source['seg_start_pos'][:]
                    if len(self.select_ids)<len(self.seg_start_pos):
                        self.seg_start_pos = self.seg_start_pos[self.select_ids]
                else:
                    self.seg_start_pos = data_source['seg_start_pos'][list(self.select_ids)]

            if 'lengths' in data_source:
                if load_all_first:
                    self.lengths = data_source['lengths'][:]
                    if len(self.select_ids)<len(self.lengths):
                        self.lengths = self.lengths[self.select_ids]
                else:
                    self.lengths = data_source['lengths'][list(self.select_ids)]
                
            #if 'label_times' in data_source:
            if load_all_first:
                self.label_times = data_source['label_times'][:]
                if len(self.select_ids)<len(self.label_times):
                    self.label_times = self.label_times[self.select_ids]
            else:
                self.label_times = data_source['label_times'][list(self.select_ids)]
            self.label_times = self.label_times.flatten()
            """
            # for each label type, only take its own label times, remove label times that is not this label type
            # for example, 'rass...camicu...' --> 'rass...'
            # this is to remove discontinuous such as rass1, rass1, rass1camicu1, rass1camicu1, rass1
            label_types = ['rass','camicu']
            # number of elements for each label type, for example
            # rass\t1900-01-01 19:00:00.0\t0\tgood --> 4 elements
            # camicu\t1900-01-01 19:00:00.0\tLauren --> 3 elements
            label_type_n_elements = [4,3]
            import pdb
            pdb.set_trace()
            li = label_types.index(self.label_type.lower())
            label_times = []
            for xx in self.label_times:
                label_elements = map(str.lower, map(str.strip, xx.split('\t')))
                mm = np.in1d(label_types, label_elements)
                if mm[li]:
                    # only take the current label type
                    label_type_id = label_elements.index(self.label_type)
                    label_elements = xx.split('\t')
                    label_times.append('\t'.join([label_elements[0]]+label_elements[label_type_id:label_type_id+label_type_n_elements[li]]))
                else:
                    raise ValueError
            self.label_times = np.array(label_times)
            """
            self.unique_label_times = self.unique_keeporder(self.label_times)
                
            if self.label_type.lower()=='camicu':
                # set y[:,0] to its RASS
                rass_times = pd.read_csv('/data/delirium/mycode_waveform_ordinal/data/rass_times.txt', sep='\t')
                # make assessments with RASS -4/-5 --> CAMICU 1
                # this is by label assessment
                for lt in self.unique_label_times:
                    lt_el = lt.split('\t')
                    patient_ids = np.where(rass_times.PatientID==lt_el[0])[0]
                    this_time = datetime.datetime.strptime(lt_el[2], '%Y-%m-%d %H:%M:%S.%f')
                    this_patient_times = np.array(map(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'), rass_times.Time[patient_ids]))
                    dt = list(map(lambda x:abs(x.total_seconds()), this_patient_times-this_time))
                    minid = np.argmin(dt)
                    if dt[minid]<600:# and rass_times.Staff[patient_ids].values[minid]!=0:
                        rass_id = patient_ids[np.argmin(dt)]
                    else:
                        #print(lt)
                        continue
                    rass = rass_times.Score[rass_id]
                
                    ids = np.where(self.label_times==lt)[0]
                    self.y[ids,0] = rass
                    if rass<=-4:
                        self.y[ids, -1] = 1
                    
            self.set_y_related()
            self.len = len(self.y)
                
            if 'drug' in data_source:
                if load_all_first:
                    self.drugs = data_source['drug'][:]
                    if len(self.select_ids)<len(self.drugs):
                        self.drugs = self.drugs[self.select_ids]
                else:
                    self.drugs = data_source['drug'][list(self.select_ids)]
                self.drug_names = data_source['drugname'][:]
                
            #if self.label_type.lower()=='camicu':
            #    self.select_ids = np.arange(self.len)

    #TODO make it decorator?
    #def validate(self, toassert=False):
    #    good = len(self)==len(self.select_ids)==len(self.y)#TODO==...
    #    good = good and self.X.ndim in [3,4]
    #    if toassert:
    #        assert good, 'Invalid Dataset object.'
    #    return good
    
    def repeat(self, times, noise_std=None):
        self.select_ids = np.repeat(self.select_ids, times, axis=0)
        self.y = np.repeat(self.y, times, axis=0)
        self.Z = np.repeat(self.Z, times, axis=0)
        #self.drugs = np.repeat(self.drugs, times, axis=0)
        self.len = len(self.select_ids)
        self.noise_std = noise_std
        self.repeated_times = times
    
    def derepeat(self):
        ids = np.arange(0, self.len, self.repeated_times)
        self.select_ids = self.select_ids[ids]
        self.y = self.y[ids]
        self.Z = self.Z[ids]
        #self.drugs = np.repeat(self.drugs, times, axis=0)
        self.len = len(self.select_ids)
        self.noise_std = None
                
    def set_X(self, X):
        #assert len(X)==self.len
        self.X = X
        self.select_ids = np.arange(len(self.X))
        if self.data_type=='specs':
            self.len, self.time_step, self.D = X.shape
            self.Dch = self.D//self.channel_num
        elif self.data_type=='eeg':
            if X.ndim==3:
                self.len, self.channel_num, self.Dch = X.shape
            else:
                self.len, self.time_step, self.channel_num, self.Dch = X.shape
            self.D = self.Dch
    
    def drop_X(self):
        # usually for save space
        del self.X
        return self
    
    def set_y_related(self, class_weight_mapping=None):
        """
        set unique_y, y_counter, K, Z, class_weight_mapping (if class_weighted)
        """
        if self.label_type.lower()=='rass':
            self.unique_y = np.sort(np.unique(self.y))
            self.K = len(self.unique_y)
            self.Z = np.ones_like(self.y).astype(float)
            self.class_weight_mapping = {yy:1. for yy in self.unique_y}
            if self.class_weighted:
                if class_weight_mapping is None:
                    self.y_counter = Counter(self.y)
                    for yy in self.unique_y:
                        self.Z[self.y==yy] = 1./self.y_counter[yy]
                    self.Z = self.Z/self.Z.mean()
                    self.class_weight_mapping = {yy:self.Z[self.y==yy][0] for yy in self.unique_y}
                else:
                    self.Z = np.array([self.class_weight_mapping[self.y[ii]] for ii in range(len(self.Z))])
                
        elif self.label_type.lower()=='camicu':
            self.K = 2
            self.unique_y = np.arange(self.K)
            self.Z = np.ones_like(self.y[:,-1]).astype(float)
            self.class_weight_mapping = {yy:1. for yy in self.unique_y}
            if self.class_weighted:
                if class_weight_mapping is None:
                    self.y_counter = Counter(self.y[:,-1])
                    for yy in self.unique_y:
                        self.Z[self.y[:,-1]==yy] = 1./self.y_counter[yy]
                    self.Z = self.Z/self.Z.mean(axis=0)
                    self.class_weight_mapping = {yy:self.Z[self.y[:,-1]==yy][0] for yy in self.unique_y}
                else:
                    self.Z = np.array([self.class_weight_mapping[self.y[ii,-1]] for ii in range(len(self.Z))])
                
        self.unique_patients = self.unique_keeporder(self.patients)
        self.unique_label_times = self.unique_keeporder(self.label_times)

    def unique_keeporder(self, x):
        _, idx = np.unique(x, return_index=True)
        return x[np.sort(idx)]

    #def fit_one_hot_encode(self, x):
    #    unique_x = self.unique_keeporder(x)
    #    mapping = {zz:i for i, zz in enumerate(unique_x)}
    #    x_enc1d = np.array(map(lambda i:mapping[i], x))
    #    # one-hot encoding
    #    encoder = LabelBinarizer().fit(x_enc1d)
    #    x_enc2d = encoder.transform(x_enc1d)
    #    return unique_x, mapping, encoder, x_enc1d, x_enc2d

    def X_astype(self, dtype):
        if hasattr(self, 'X') and self.X.dtype!=dtype:
            self.X = self.X.astype(dtype)
        return self

    def fliplr(self):
        if len(self.X)>len(self.select_ids):
            self.X = self.X[self.select_ids]
            self.select_ids = np.arange(len(self.X))
        if self.X.ndim==3:
            self.X = np.tile(self.X, (2,1,1))
            self.X[self.len:, [0,1]] = self.X[self.len:, [1,0]]
            #self.X[self.len:, [2,3]] = -self.X[self.len:, [2,3]]
        elif self.X.ndim==4:
            self.X = np.tile(self.X, (2,1,1,1))
            self.X[self.len:, :, [0,1]] = self.X[self.len:, :, [1,0]]
            #self.X[self.len:, :, [2,3]] = -self.X[self.len:, :, [2,3]]
        self.y = np.r_[self.y, self.y]
        self.Z = np.r_[self.Z, self.Z]
        self.seg_start_pos = np.r_[self.seg_start_pos, self.seg_start_pos]
        self.patients = np.r_[self.patients, self.patients]
        self.label_times = np.r_[self.label_times, self.label_times]
        if hasattr(self, 'lengths'):
            self.lengths = np.r_[self.lengths, self.lengths]
        #TODO self.drugs = np.r_[self.drugs, self.drugs]
        
        self.select_ids = np.r_[self.select_ids, self.select_ids+self.len]
        self.len = len(self.select_ids)
        return self

    def defliplr(self):
        assert len(self.X)==len(self.select_ids) and self.len%2==0
        N = self.len//2
        self.X = self.X[:N]
        self.y = self.y[:N]
        self.Z = self.Z[:N]
        self.patients = self.patients[:N]
        self.seg_start_pos = self.seg_start_pos[:N]
        self.label_times = self.label_times[:N]
        if hasattr(self, 'lengths'):
            self.lengths = self.lengths[:N]
        #TODO self.drugs = self.drugs[:N]
        self.len = len(self.X)
        self.select_ids = np.arange(len(self.X))
        return self
        
    def shorten(self, amount=2):
        """
        Half the time step for recurrent data type
        """
        # check if it is recurrent data type
        if self.seg_start_pos.ndim!=2:
            raise TypeError('Cannot shorten non-recurrent dataset.')
        self.shorten_amount = amount
            
        if len(self.X)>len(self.select_ids):
            self.X = self.X[self.select_ids]
            self.select_ids = np.arange(len(self.X))
        self.shorten_padding = int(np.ceil(self.X.shape[1]*1./self.shorten_amount))*self.shorten_amount-self.X.shape[1]
        if self.shorten_padding>0:
            self.X = np.concatenate([self.X, np.zeros((self.X.shape[0],self.shorten_padding)+self.X.shape[2:])], axis=1)
            self.seg_start_pos = np.concatenate([self.seg_start_pos, -np.ones((self.seg_start_pos.shape[0],self.shorten_padding))], axis=1)
        l2 = self.X.shape[1]//self.shorten_amount
        
        self.X = np.concatenate([self.X[:,l2*ii:l2*(ii+1)] for ii in range(self.shorten_amount)], axis=0)
        self.seg_start_pos = np.concatenate([self.seg_start_pos[:,l2*ii:l2*(ii+1)] for ii in range(self.shorten_amount)], axis=0)
        self.y = np.concatenate([self.y]*self.shorten_amount, axis=0)
        self.Z = np.concatenate([self.Z]*self.shorten_amount, axis=0)
        self.patients = np.concatenate([self.patients]*self.shorten_amount, axis=0)
        self.label_times = np.concatenate([self.label_times]*self.shorten_amount, axis=0)
        #TODO drugs, ...
        self.select_ids = np.concatenate([self.select_ids+self.len*ii for ii in range(self.shorten_amount)], axis=0)
        if hasattr(self, 'lengths'):
            self.lengths = np.maximum(np.minimum(
                                np.repeat(np.arange(1,self.shorten_amount+1)*l2, self.len),
                                np.tile(self.lengths, self.shorten_amount)
                            )-np.repeat(np.arange(self.shorten_amount)*l2, self.len), 0)
        if self.X.ndim==4:
            self.len, self.time_step, self.channel_num, self.Dch = self.X.shape
            self.D = self.Dch
        elif self.X.ndim==3:
            self.len, self.time_step, self.D = self.X.shape
            self.Dch = self.D
        return self
        
    #TODO def deshorten(self):
    #    if self.shorten_amount is None:
    #        return self
    #    self.shorten_amount = None
    #    return self
    
    def flatten_channel(self):
        """
        N x C x D --> (NxC) x 1 x D
        """
        assert self.data_type=='eeg' and self.X.ndim==3
                
        if len(self.X)>len(self.select_ids):
            self.X = self.X[self.select_ids]
        repeat_num = self.X.shape[1]
        self.X = self.X.reshape(self.X.shape[0]*repeat_num, 1, self.X.shape[2])
        self.select_ids = np.arange(len(self.X))
        self.len, self.channel_num, self.Dch = self.X.shape
        self.D = self.Dch
        self.y = np.repeat(self.y, repeat_num, axis=0)
        self.Z = np.repeat(self.Z, repeat_num, axis=0)
        self.patients = np.repeat(self.patients, repeat_num, axis=0)
        self.seg_start_pos = np.repeat(self.seg_start_pos, repeat_num, axis=0)
        self.label_times = np.repeat(self.label_times, repeat_num, axis=0)
        if hasattr(self, 'lengths'):
            self.lengths = np.repeat(self.lengths, repeat_num, axis=0)
        self.drugs = np.repeat(self.drugs, repeat_num, axis=0)
        if hasattr(self, 'augmented'):
            self.augmented = np.repeat(self.augmented, repeat_num, axis=0)
        return self
    
    def longer_window_size(self):
        """
        Make longer window size, 4s --> 8s
        """
        if self.data_type=='eeg' and self.X.ndim==3:
            cat_ids1 = []
            cat_ids2 = []
            seg_start_pos = self.seg_start_pos.tolist()
            for lt in self.unique_label_times:
                ids = np.where(self.label_times==lt)[0]
                aa = self.seg_start_pos[ids]
                bb = aa + 1000
                cat_ids1.extend(ids[np.in1d(bb, aa)])
                cat_ids2.extend(ids[np.in1d(aa, bb)])
                
            if len(self.X)>len(self.select_ids):
                self.X = self.X[self.select_ids]
                
            self.X = np.concatenate([self.X[cat_ids1], self.X[cat_ids2]], axis=2)
            _, self.channel_num, self.Dch = self.X.shape
            self.select_ids = np.arange(len(self.X))
            self.y = self.y[cat_ids1]
            self.Z = self.Z[cat_ids1]
            self.patients = self.patients[cat_ids1]
            self.seg_start_pos = self.seg_start_pos[cat_ids1]
            self.label_times = self.label_times[cat_ids1]
            if hasattr(self, 'lengths'):
                self.lengths = self.lengths[cat_ids1]
            self.drugs = self.drugs[cat_ids1]
            self.len = len(self.X)
            self.set_y_related()
        elif self.data_type=='eeg' and self.X.ndim==4:
            if len(self.X)>len(self.select_ids):
                self.X = self.X[self.select_ids]
                self.select_ids = np.arange(len(self.X))
            self.X = self.X.transpose(0,2,1,3).reshape(self.X.shape[0],self.X.shape[2],self.X.shape[1]//2,self.X.shape[3]*2).transpose(0,2,1,3)
            _, self.time_step, self.channel_num, self.Dch = self.X.shape
            self.D = self.Dch
            self.seg_start_pos = self.seg_start_pos[:,::2]
        return self
            
    def summary(self, suffix='', items=None):
        print('\n'+suffix)
        if items is None:
            items = ['patients','records','assessments','samples','labels','Z','drugs']
        if 'patients' in items and hasattr(self, 'unique_patients'):
            patient_num = len(self.unique_patients)
            print('patient number %d'%patient_num)
        if 'records' in items and hasattr(self, 'unique_records'):
            record_num = len(self.unique_records)
            print('record number %d'%record_num)
        if 'assessments' in items and hasattr(self, 'unique_assessments'):
            assessment_num = len(self.unique_assessments)
            print('assessment number %d'%assessment_num)
        if 'samples' in items and hasattr(self, 'len'):
            print('sample number %d'%self.len)
        if 'labels' in items and hasattr(self, 'y'):
            print('labels %s'%self.y_counter)
        #if 'Z' in items and hasattr(self, 'Z'):
        #    print('Zs %s'%Counter(self.Z.flatten()))
        #if 'drugs' in items and hasattr(self, 'drugs'):
        #    print('drugs: %s'%self.drug_names)
    
    def __iter__(self):
        return self.__dict__.iteritems()

    def __len__(self):
        return self.len
        
    def __getitem__(self, id_):
        if type(id_)==int:
            idx = [id_]
        else:
            idx = id_
        X = self.X[self.select_ids[idx]]
        y = self.y[idx]
        #drugs = self.drugs[idx]
        Z = self.Z[idx]
        if hasattr(self, 'lengths'):
            lengths = self.lengths[idx]
        
        if self.fliplr_prob:
            # flip left and right electrodes, eeg_channels = ['Fp1-F7','Fp2-F8']#,'Fp1-Fp2','F7-F8']
            flipids = np.where(np.random.rand(len(idx))>=0.5)[0]
            if len(flipids)>0:
                if self.data_type=='specs':
                    raise NotImplemented('fliplr for specs?')
                    #TODO ndim=3/4
                    N = X.shape[0]
                    #X = X.reshape(N,4,self.Dch)[:,[1,0,2,3]].reshape(N,4*self.Dch)
                    X = X.reshape(N,2,self.Dch)[:,[1,0]].reshape(N,2*self.Dch)
                else:
                    if X.ndim==3:
                        X[flipids,[0,1]] = X[flipids,[1,0]]
                        #X[flipids,[2,3]] = -X[flipids,[2,3]]
                    elif X.ndim==4:
                        X[flipids,:,[0,1]] = X[flipids,:,[1,0]]
                        #X[flipids,:,[2,3]] = -X[flipids,:,[2,3]]
        
        #if self.noise_std is not None:
        #    X = X + np.random.randn(*X.shape)*self.noise_std
        if type(id_)==int:
            X = X[0]
            Z = Z[0]
            y = y[0]
            idx = idx[0]
            if hasattr(self, 'lengths'):
                lengths = lengths[0]
            
        res = {'X': X.astype('float32'), 'Z':Z.astype('float32'), 'y': y.astype('float32'), 'ids':idx}#, 'drugs': drugs.astype('float32')}
        if hasattr(self, 'lengths'):
            res['L'] = lengths#.astype(long)
            
        return res


def copy_dataset(dataset, copy_X=False):
    # TODO def copy(self)
    newdataset = MyDataset(copy.deepcopy(dataset.input_path),
                    copy.deepcopy(dataset.label_type),
                    data_type=copy.deepcopy(dataset.data_type),
                    inmemory=copy.deepcopy(dataset.inmemory),
                    fliplr_prob=copy.deepcopy(dataset.fliplr_prob),
                    class_weighted=copy.deepcopy(dataset.class_weighted),
                    label_mapping=copy.deepcopy(dataset.label_mapping),
                    really_load=False)#variable_names=None,
    
    if hasattr(dataset, 'select_ids'):
        newdataset.select_ids = np.array(dataset.select_ids, copy=True)
    if hasattr(dataset, 'repeated_times'):
        newdataset.repeated_times = copy.deepcopy(dataset.repeated_times)
    if hasattr(dataset, 'noise_std'):
        newdataset.noise_std = copy.deepcopy(dataset.noise_std)
    if hasattr(dataset, 'shorten_amount'):
        newdataset.shorten_amount = copy.deepcopy(dataset.shorten_amount)
    if hasattr(dataset, 'shorten_padding'):
        newdataset.shorten_padding = copy.deepcopy(dataset.shorten_padding)
    if hasattr(dataset, 'len'):
        newdataset.len = copy.deepcopy(dataset.len)
    if hasattr(dataset, 'inv_label_mapping'):
        newdataset.inv_label_mapping = copy.deepcopy(dataset.inv_label_mapping)
    #if hasattr(dataset, 'time_step'):
    #    newdataset.time_step = copy.deepcopy(dataset.time_step)
    if hasattr(dataset, 'drug_names'):
        newdataset.drug_names = np.array(dataset.drug_names, copy=True)
    if hasattr(dataset, 'freqs'):
        newdataset.freqs = np.array(dataset.freqs, copy=True)

    if hasattr(dataset, 'X'):
        if copy_X:
            newdataset.X = np.array(dataset.X, copy=True) # this can cost many memory
        else:
            newdataset.X = dataset.X # use the same memory, but index with different select_ids
        if newdataset.data_type=='specs':
            _, newdataset.time_step, newdataset.D = newdataset.X.shape
            newdataset.channel_num = 2####
            newdataset.Dch = newdataset.D//newdataset.channel_num
        elif newdataset.data_type=='eeg':
            if newdataset.X.ndim==3:
                _, newdataset.channel_num, newdataset.Dch = newdataset.X.shape
            else:
                _, newdataset.time_step, newdataset.channel_num, newdataset.Dch = newdataset.X.shape
            newdataset.D = newdataset.Dch
        
    if hasattr(dataset, 'y'):
        newdataset.y = np.array(dataset.y, copy=True)
    
    if hasattr(dataset, 'drugs'):
        newdataset.drugs = np.array(dataset.drugs, copy=True)

    if hasattr(dataset, 'label_times'):
        newdataset.label_times = np.array(dataset.label_times, copy=True)
        
    if hasattr(dataset, 'patients'):
        newdataset.patients = np.array(dataset.patients, copy=True)
        
    if hasattr(dataset, 'seg_start_pos'):
        newdataset.seg_start_pos = np.array(dataset.seg_start_pos, copy=True)

    if hasattr(dataset, 'lengths'):
        newdataset.lengths = np.array(dataset.lengths, copy=True)

    if hasattr(dataset, 'augmented'):
        newdataset.augmented = np.array(dataset.augmented, copy=True)
    newdataset.set_y_related()

    return newdataset


def slice_dataset(dataset, ids):#, fliplr=None):
    # TODO def __item__
    assert len(ids)==len(np.unique(ids))
    
    newdataset = copy_dataset(dataset, copy_X=False)
    #if fliplr is not None:
    #    newdataset.fliplr = fliplr

    #if hasattr(dataset, 'X'):
    #    newdataset.X = newdataset.X[ids]
    #    newdataset.len = len(newdataset.X)
    if hasattr(dataset, 'select_ids'):
        newdataset.select_ids = newdataset.select_ids[ids]
        newdataset.len = len(newdataset.select_ids)
        
    if hasattr(dataset, 'y'):
        newdataset.y = newdataset.y[ids]
    
    if hasattr(dataset, 'drugs'):
        newdataset.drugs = dataset.drugs[ids]

    if hasattr(dataset, 'label_times'):
        newdataset.label_times = newdataset.label_times[ids]
        
    if hasattr(dataset, 'patients'):
        newdataset.patients = newdataset.patients[ids]
        
    if hasattr(dataset, 'seg_start_pos'):
        newdataset.seg_start_pos = newdataset.seg_start_pos[ids]

    if hasattr(dataset, 'lengths'):
        newdataset.lengths = newdataset.lengths[ids]

    if hasattr(dataset, 'augmented'):
        newdataset.augmented = newdataset.augmented[ids]
    
    newdataset.set_y_related()

    return newdataset


def combine_dataset(d1, d2):#, return_comb_ids=False):
    # TODO def __add__
    newdataset = copy_dataset(d1, copy_X=False)
    if hasattr(d1, 'fliplr_prob') and hasattr(d2, 'fliplr_prob'):
        newdataset.fliplr_prob = d1.fliplr_prob or d2.fliplr_prob
    elif hasattr(d2, 'fliplr_prob'):
        newdataset.fliplr_prob = d2.fliplr_prob
    else:
        newdataset.fliplr_prob = False

    #if hasattr(d1, 'X') and hasattr(d2, 'X'):
    #    newdataset.X = np.r_[d1.X, d2.X]
    #    newdataset.len = len(d1)+len(d2)
    if hasattr(d1, 'select_ids') and hasattr(d2, 'select_ids'):
        #assert len(set(d1.select_ids) & set(d2.select_ids))==0
        newdataset.select_ids = np.r_[d1.select_ids, d2.select_ids+len(d1.X)]
        newdataset.len = len(newdataset.select_ids)
        
    if hasattr(d1, 'y') and hasattr(d2, 'y'):
        newdataset.y = np.r_[d1.y, d2.y]
    
    if hasattr(d1, 'drugs') and hasattr(d2, 'drugs'):
        newdataset.drugs = np.r_[d1.drugs, d2.drugs]

    if hasattr(d1, 'label_times') and hasattr(d2, 'label_times'):
        newdataset.label_times = np.r_[d1.label_times, d2.label_times]
    
    if hasattr(d1, 'patients') and hasattr(d2, 'patients'):
        newdataset.patients = np.r_[d1.patients, d2.patients]
        
    if hasattr(d1, 'records') and hasattr(d2, 'records'):
        newdataset.records = np.r_[d1.records, d2.records]

    if hasattr(d1, 'assessments') and hasattr(d2, 'assessments'):
        newdataset.assessments = np.r_[d1.assessments, d2.assessments]
        
    if hasattr(d1, 'seg_start_pos') and hasattr(d2, 'seg_start_pos'):
        newdataset.seg_start_pos = np.r_[d1.seg_start_pos, d2.seg_start_pos]
        
    if hasattr(d1, 'lengths') and hasattr(d2, 'lengths'):
        newdataset.lengths = np.r_[d1.lengths, d2.lengths]
        
    if hasattr(d1, 'augmented') and hasattr(d2, 'augmented'):
        newdataset.augmented = np.r_[d1.augmented, d2.augmented]
    newdataset.set_y_related()

    return newdataset


def make_test_dataset(dataset, min_len=None):
    """
    for each assessment, create a long sequence, possibly with different lengths but padded
    """
    newdataset = copy_dataset(dataset, copy_X=False)
    if len(dataset.select_ids)<len(dataset.X):
        newdataset.X = dataset.X[dataset.select_ids]
        newdataset.select_ids = np.arange(len(newdataset.X))
        
    #seg_interval = 1000
    L = -np.inf
    seg_interval = np.diff(newdataset.seg_start_pos,axis=1).min()
    for lt in newdataset.unique_label_times:
        this_ids = np.where(newdataset.label_times==lt)[0]
        this_seg_start_pos = newdataset.seg_start_pos[this_ids]
        ll = this_seg_start_pos.max()-this_seg_start_pos.min()
        if ll>L: L = ll
    L = int(np.ceil(L*1./seg_interval))+1
        
    ys = []
    seg_start_poss = []
    Xs = []
    lengths = []
    patients = []
    for lt in tqdm(newdataset.unique_label_times, leave=False):
        this_ids = np.where(newdataset.label_times==lt)[0]
        
        # y
        if newdataset.y.ndim==2:
            assert len(np.unique(newdataset.y[this_ids,-1]))==1
        else:
            assert len(np.unique(newdataset.y[this_ids]))==1
            
        
        # seg_start_pos
        min_pos = newdataset.seg_start_pos[this_ids].min()
        seg_start_pos = np.arange(min_pos, min_pos+L*seg_interval, seg_interval)
        
        # X
        X = np.zeros((L, dataset.channel_num, dataset.Dch), dtype=newdataset.X.dtype)+np.nan
        """
        initial_seg_start_pos = newdataset.seg_start_pos[this_ids][:,0].tolist()
        for ii in range(len(X)):
            if np.all(X[ii]==0) and seg_start_pos[ii] in initial_seg_start_pos:
                id_ = initial_seg_start_pos.index(seg_start_pos[ii])
                toplace_ids = np.where(np.in1d(seg_start_pos, newdataset.seg_start_pos[this_ids][id_]))[0]
                X[toplace_ids] = newdataset.X[this_ids][id_]
        """
        #good = True
        for this_id in this_ids:
            ids = np.where(np.in1d(seg_start_pos, newdataset.seg_start_pos[this_id]))[0]
            if len(ids)!=newdataset.X.shape[1]:
                #print(lt)
                #good = False
                #break
                continue
            X[ids] = newdataset.X[this_id]
        #if not good:
        #    continue
        
        # length: [x x nan x x x nan nan nan nan ...] --> len=6
        for len_ in range(L-1,-1,-1):
            if not np.all(np.isnan(X[len_])):
                break
        len_ += 1
        # if nan in middle, not at the end, skip
        if np.any(np.isnan(X[:len_])):
            continue
            
        lengths.append(len_)#len(good_ids))
            
        #good_ids = np.where(~np.all(X==0, axis=(1,2)))[0]
        #if len(good_ids)<len(X):
        #    X = np.r_[X[good_ids], np.zeros((len(X)-len(good_ids), dataset.channel_num, dataset.Dch), dtype=newdataset.X.dtype)]
        Xs.append(X)
            
        # patients
        assert len(np.unique(newdataset.patients[this_ids]))==1
        patients.append(newdataset.patients[this_ids][0])
        
        # TODO drugs
        
        ys.append(newdataset.y[this_ids][0])
        seg_start_poss.append(seg_start_pos)
        
    newdataset.lengths = np.array(lengths)
    newdataset.label_times = np.array(newdataset.unique_label_times, copy=True)
    newdataset.y = np.array(ys)
    newdataset.seg_start_pos = np.array(seg_start_poss)
    newdataset.X = np.array(Xs)
    newdataset.X[np.isnan(newdataset.X)] = 0.
    newdataset.patients = np.array(patients)
    newdataset.fliplr_prob = False
    if min_len is not None:
        goodids = np.where(newdataset.lengths>min_len)[0]
    else:
        goodids = np.arange(len(newdataset.lengths))
    if len(goodids)<len(newdataset.lengths):
        newdataset.lengths = newdataset.lengths[goodids]
        newdataset.label_times = newdataset.label_times[goodids]
        newdataset.y = newdataset.y[goodids]
        newdataset.seg_start_pos = newdataset.seg_start_pos[goodids]
        newdataset.patients = newdataset.patients[goodids]
        #newdataset.select_ids = good_ids
        newdataset.X = newdataset.X[goodids]
    newdataset.set_y_related()  # deals Z
    newdataset.len = len(newdataset.y)
    newdataset.select_ids = np.arange(newdataset.len)
    
    actual_min_len = min(newdataset.lengths.max(),newdataset.X.shape[1])
    newdataset.X = newdataset.X[:,:actual_min_len]
    newdataset.seg_start_pos = newdataset.seg_start_pos[:,:actual_min_len]
                
    if newdataset.data_type=='specs':
        newdataset.h5_len, newdataset.time_step, newdataset.D = newdataset.X.shape
        newdataset.Dch = newdataset.D//newdataset.channel_num
    elif newdataset.data_type=='eeg':
        if newdataset.X.ndim==3:
            _, newdataset.channel_num, newdataset.Dch = newdataset.X.shape
        else:
            _, newdataset.time_step, newdataset.channel_num, newdataset.Dch = newdataset.X.shape
        newdataset.D = newdataset.Dch
    
    return newdataset
    

def change_window_setting(dataset, length, step, output_type):
    old_length = 4 # [s]TODO newdataset.window_length_second
    length2 = length//old_length
    step2 = step//old_length
    
    if output_type=='ff':
        X = []
        y = []
        #drugs = []
        label_times = []
        patients = []
        seg_start_pos = []
        augmented = []
        newdataset = make_test_dataset(dataset)
        for i in range(len(newdataset.X)):
            ids = np.arange(0,len(newdataset.X[i])-length2+1, step2)
            X.extend([newdataset.X[i, j:j+length2] for j in ids])
            y.extend([newdataset.y[i]]*len(ids))
            label_times.extend([newdataset.label_times[i]]*len(ids))
            patients.extend([newdataset.patients[i]]*len(ids))
            seg_start_pos.extend(newdataset.seg_start_pos[i, ids])
            if hasattr(newdataset, 'augmented'):
                augmented.extend([newdataset.augmented[i]]*len(ids))
        X = np.array(X).transpose(0,2,1,3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        
        newdataset.set_X(X)
        newdataset.y = np.array(y)
        newdataset.label_times = np.array(label_times)
        newdataset.patients = np.array(patients)
        newdataset.seg_start_pos = np.array(seg_start_pos)
        if hasattr(newdataset, 'augmented'):
            newdataset.augmented = np.array(augmented)
                
    elif output_type=='rnn':
        newdataset = copy_dataset(dataset, copy_X=True)
        if len(newdataset.select_ids)<len(newdataset.X):
            newdataset.X = newdataset.X[newdataset.select_ids]
            newdataset.select_ids = np.arange(len(newdataset.X))
        X = []
        seg_start_pos = []
        for i in range(len(newdataset.X)):
            ids = np.arange(0,len(newdataset.X[i])-length2+1, step2)
            X.append([newdataset.X[i, j:j+length2] for j in ids])
            seg_start_pos.append(newdataset.seg_start_pos[i, ids])
        X = np.array(X).transpose(0,1,3,2,4)
        newdataset.set_X(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1))
        newdataset.seg_start_pos = np.array(seg_start_pos)
    else:
        raise NotImplementedError('Unknown output type %s'%output_type)
        
    if hasattr(newdataset, 'lengths'):
        del newdataset.lengths
    newdataset.select_ids = np.arange(len(newdataset.X))
    newdataset.len = len(newdataset.X)
    newdataset.set_y_related()
    
    return newdataset
