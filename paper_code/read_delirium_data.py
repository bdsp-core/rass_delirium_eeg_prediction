#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import numpy as np
import scipy.io as sio
import h5py
import hdf5storage as hs

SECONDS_IN_DAY = 86400.


def datenum(date_str, format_, return_seconds=False):
    # emulates datenum in matlab
    days = (datetime.datetime.strptime(date_str, format_)-datetime.datetime(1,1,1,0,0,0)).total_seconds()*1./SECONDS_IN_DAY+367.
    if return_seconds:
        return days*SECONDS_IN_DAY
    else:
        return days
        

def numdate(seconds, format_='%Y-%m-%d %H:%M:%S.%f'):
    # emulates numdate in matlab
    return datetime.datetime.utcfromtimestamp(seconds-datenum('1970-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f', return_seconds=True)).strftime(format_)


def read_delirium_mat(fn, channel_names=None, with_data=True):
    """
    fn: MAT file path
    """
    try:
        res = sio.loadmat(fn, variable_names=['Fs','labels'])
    except Exception:
        res = hs.loadmat(fn, variable_names=['Fs','labels'])
    
    if 'Fs' not in res or 'labels' not in res:
        raise TypeError('Incomplete data in %s'%fn)   
    
    if channel_names is not None:
        channel_names = map(lambda x:x.lower(), channel_names)
    labels = map(lambda x:x.tolist()[0], res['labels'].flatten())
    Fs = round(res['Fs'][0,0]*1000)/1000.
         
    if len(labels)!=5:
        raise TypeError('Not 5 channels in %s'%fn)
    if channel_names is not None and np.any(map(lambda i:channel_names[i] not in labels[i].lower(), range(len(labels)))):
        raise TypeError('Incorrect channels in %s'%fn)
    
    myres = {'labels':labels, 'Fs':Fs}
    if with_data:
        try:
            res = sio.loadmat(fn, variable_names=['data'])
        except Exception:
            res = hs.loadmat(fn, variable_names=['data'])
        if 'data' not in res:
            raise TypeError('Incomplete data in %s'%fn) 
        data = res['data']
        
        if data.shape[0]!=5:
            raise TypeError('Not 5 channels in %s'%fn)
        myres['data'] = data
        data_length = data.shape[1]
    else:
        try:
            data_length = filter(lambda x:x[0]=='data',sio.whosmat(fn))[0][1][1]
        except Exception:
            with h5py.File(fn,'r') as ff:
                data_length = ff['data'].shape[1]
        
    return myres
        
            
if __name__=='__main__':
    res = read_delirium_mat('/media/mad3/SEDATION_STUDY/SEDATION_WORKING_DATA/CopiesOfPrimarySedationData/icused20/eeg/matlab/M5060601_140202_1447-seg_converted.edf.mat')
    print(res.keys())
    print(res['data'].shape)
    
