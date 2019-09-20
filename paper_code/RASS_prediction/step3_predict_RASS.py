import os
import pickle
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import torch as th
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
import sys
sys.path.insert(0, r'myml_lib')
from dataset import *
from experiment import *
from mymodels.eegnet_feedforward import EEGNet_CNN
from mymodels.eegnet_recurrent import EEGNet_RNN, ModuleSequence


tosave = True
n_gpu = 1
label_type = 'rass'
cv_method = '10-fold'
random_state = 10
batch_size = 16
lr = 0.001
max_epoch = 10
data_type = 'eeg'

if label_type == 'rass':
    label_mapping = {-5:0,-4:1,-3:2,-2:3,-1:4,0:5}
    loss_function = 'ordinal'
elif label_type == 'camicu':
    label_mapping = None
    loss_function = 'bin'
else:
    raise ValueError('Unknown label type %s'%label_type)


def generate_tr_va_te(dataset, method, label_mapping=None, random_state=None):
    """
    genertate tr, va and te label_times
    """
    foldids = []; trids = []; vaids = []; teids = []
    unique_patients = np.array(dataset.unique_patients, copy=True)

    if method=='10-fold':
        if len(unique_patients)<10:
            raise ValueError('len(unique_patients)<10')
        np.random.seed(random_state)
        np.random.shuffle(unique_patients)
        allfolds = np.array_split(unique_patients, 10)
        foldids = []
        for k in range(10):
            foldids.append('fold %d'%(k+1,))
            patients_te = allfolds[k]
            patients_tr = np.concatenate(allfolds[:k]+allfolds[k+1:])
            # split label_times within tr to get va
            label_times = np.unique(dataset.label_times[np.in1d(dataset.patients, patients_tr)])
            np.random.seed(random_state+k+1)
            np.random.shuffle(label_times)
            trids.append(label_times[:len(label_times)//10*9])
            vaids.append(label_times[len(label_times)//10*9:])
            teids.append(np.unique(dataset.label_times[np.in1d(dataset.patients, patients_te)]))

    else:
        raise NotImplementedError(method)
        
    assert len(trids)==len(vaids)==len(teids)
    for i in range(len(trids)):
        assert len(set(trids[i]) & set(vaids[i]) & set(teids[i]))==0
        
    return foldids, trids, vaids, teids


def report_performance(y, yp, prefix=''):
    rmse = np.sqrt(np.mean((y-yp)**2))
    mae = np.mean(np.abs(y-yp))
    pn1acc = np.mean(np.abs(y-yp)<=1)
    
    msg = '%s RMSE = %g\tMAE = %g\t<=+/-1 acc = %g'%(prefix, rmse, mae, pn1acc)
    print(msg)


if __name__=='__main__':

    ## read all data
    data_path_ff = '/data/delirium/eeg_segs_%s_feedforward_w4s2.h5'%label_type
    data_path_rnn = '/data/delirium/eeg_segs_%s_recurrent_w4s2_L9.5min.h5'%label_type

    dall_ff = MyDataset(data_path_ff, label_type, data_type=data_type, label_mapping=label_mapping, class_weighted=True)
    dall_rnn = MyDataset(data_path_rnn, label_type, data_type=data_type, label_mapping=label_mapping, class_weighted=True)
    
    # remove bad quality patients
    bad_quality_patients = ['icused14', 'icused29', 'icused44', 'icused52', 'icused69', 'icused98', 'icused122', 'icused125', 'icused185', 'icused199']
    print('%d patients removed due to bad signal quality'%len(bad_quality_patients))
    select_mark = ~np.in1d(dall_rnn.patients, bad_quality_patients)
    dall_rnn = slice_dataset(dall_rnn, np.where(select_mark)[0])
    select_mark = ~np.in1d(dall_ff.patients, bad_quality_patients)
    dall_ff = slice_dataset(dall_ff, np.where(select_mark)[0])

    """
    # to generate database summary table
    unique_patients=np.unique(dall_rnn.patients)
    unique_patients=np.array(sorted(unique_patients,key=lambda x:int(x[6:])))
    aa=pd.read_csv('../data/demographics.csv',sep=',')
    patients2 = aa['PatientID'].values.tolist()
    ids=[patients2.index(x) for x in unique_patients]
    df=aa.iloc[ids].reset_index(drop=True)
    np.percentile(df['APACHEII'],(25,50,75))
    
    import datetime
    ids=np.where((~df['ICUAdmission'].isna())&(~df['ICUDischarge'].isna()))[0]
    icuadmission=np.array([datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f') for x in df.iloc[ids].ICUAdmission])
    icudischarge=np.array([datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f') for x in df.iloc[ids].ICUDischarge])
    icudays=np.array([x.total_seconds()/3600./24 for x in icudischarge-icuadmission])
    """
    
    K = dall_rnn.K
    dall_ff.summary(suffix='feedfoward all')
    dall_rnn.summary(suffix='recurrent all')

    ## generate tr, va, te folds
    
    np.random.seed(random_state+10)
    folds_path = 'RASS_folds_info_%s.pickle'%cv_method
    if os.path.exists(folds_path):
        with open(folds_path,'rb') as ff:
            foldnames, tr_label_timess, va_label_timess, te_label_timess = pickle.load(ff)
    else:
        foldnames, tr_label_timess, va_label_timess, te_label_timess = generate_tr_va_te(dall_rnn, cv_method, random_state=random_state)
        with open(folds_path,'wb') as ff:
            pickle.dump([foldnames, tr_label_timess, va_label_timess, te_label_timess], ff, protocol=2)

    ## train
    for fi, fold, tr_label_times, va_label_times, te_label_times in zip(range(len(foldnames)), foldnames, tr_label_timess, va_label_timess, te_label_timess):
        print('\n########## [%d/%d] %s ##########\n'%(fi+1, len(foldnames), fold))
        cnn_model_path = 'models/model_RASS_cnn_%s.pth'%fold
        rnn1_model_path = 'models/model_RASS_rnn1_%s.pth'%fold
        rnn2_model_path = 'models/model_RASS_rnn2_%s.pth'%fold
        result_path = 'results/results_RASS_%s.pickle'%fold
        if os.path.exists(result_path):
            continue
        
        # step 1: train CNN

        exp = Experiment(model=EEGNet_CNN(loss_function, K),
                    batch_size=batch_size, max_epoch=max_epoch, lr=lr,
                    loss_function=loss_function, clip_weight=False,
                    n_gpu=n_gpu, verbose=True, random_state=random_state)

        dtr_ff = slice_dataset(dall_ff, np.where(np.in1d(dall_ff.label_times, tr_label_times))[0])
        dva_ff = slice_dataset(dall_ff, np.where(np.in1d(dall_ff.label_times, va_label_times))[0])
        dte_ff = slice_dataset(dall_ff, np.where(np.in1d(dall_ff.label_times, te_label_times))[0])
        dtr_ff.summary(suffix='feedfoward tr')
        dva_ff.summary(suffix='feedfoward va')
        dte_ff.summary(suffix='feedfoward te')
    
        print('Model CNN parameters: %d'%exp.model.n_param)
        dtr_ff.fliplr_prob=True
        exp.fit(dtr_ff, Dva=dva_ff, init=False)
        dtr_ff.fliplr_prob = False
        if tosave:
            exp.save(cnn_model_path)
        #exp.load(cnn_model_path)
        #print('model loaded from %s'%cnn_model_path)

        del dtr_ff
        del dva_ff
        del dall_ff

        # step 2: get CNN outputs

        dtr = slice_dataset(dall_rnn, np.where(np.in1d(dall_rnn.label_times, tr_label_times))[0])
        dva = slice_dataset(dall_rnn, np.where(np.in1d(dall_rnn.label_times, va_label_times))[0])
        dte = slice_dataset(dall_rnn, np.where(np.in1d(dall_rnn.label_times, te_label_times))[0])
        dtr.summary(suffix='recurrent tr')
        dva.summary(suffix='recurrent va')
        dte.summary(suffix='recurrent te')
        
        exp.batch_size = 64
        dtr.set_X(exp.predict(dtr, output_id=1, return_last=False))
        dva.set_X(exp.predict(dva, output_id=1, return_last=False))
        dte.set_X(exp.predict(dte, output_id=1, return_last=False))
        L = dtr.X.shape[2]

        # step 3: train RNN using CNN outputs
        
        exp = Experiment(batch_size=batch_size, max_epoch=max_epoch*5, lr=lr,
                    loss_function=loss_function, clip_weight=0.1,
                    n_gpu=n_gpu, verbose=True, random_state=random_state)
        exp.model = EEGNet_RNN(loss_function, L, K, model_type='lstm', rnn_layer_num=2, rnn_hidden_num=16, rnn_dropout=0)
        print('Model RNN parameters: %d'%exp.model.n_param)
        exp.fit(dtr, Dva=dva, return_last=False)
        if tosave:
            exp.save(rnn1_model_path)
        #exp.load(rnn1_model_path)
        #print('model loaded from %s'%rnn1_model_path)
        
        dtr.set_X(np.expand_dims(dtr.X, axis=2))
        dva.set_X(np.expand_dims(dva.X, axis=2))
        dte.set_X(np.expand_dims(dte.X, axis=2))
        dtr2 = make_test_dataset(dtr)#, min_len=1200//4)  # min length 20min (/4 due to step=4s)
        dva2 = make_test_dataset(dva)#, min_len=1200//4)
        dte2 = make_test_dataset(dte)#, min_len=1200//4)
        dtr2.set_X(dtr2.X[:,:,0,:])
        dva2.set_X(dva2.X[:,:,0,:])
        dte2.set_X(dte2.X[:,:,0,:])
        dtr2.summary(suffix='recurrent tr2')
        dva2.summary(suffix='recurrent va2')
        dte2.summary(suffix='recurrent te2')
        
        exp.batch_size = 64
        dtr2.set_X(exp.predict(dtr2, output_id=1, return_last=False))
        dva2.set_X(exp.predict(dva2, output_id=1, return_last=False))
        dte2.set_X(exp.predict(dte2, output_id=1, return_last=False))
        
        # step 4: train RNN on 1h segment

        exp = Experiment(batch_size=4, max_epoch=max_epoch*20, lr=lr/5.,
                    loss_function=loss_function, clip_weight=0.1,# stateful=True,
                    n_gpu=n_gpu, verbose=True, random_state=random_state)
        exp.model = EEGNet_RNN(loss_function, dtr2.X.shape[2], K, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=8, rnn_dropout=0.2)
        print('Model RNN2 parameters: %d'%exp.model.n_param)
        exp.fit(dtr2, Dva=dva2, return_last=False)
        if tosave:
            exp.save(rnn2_model_path)
        #exp.load(rnn2_model_path)
        #print('model loaded from %s'%rnn2_model_path)

        # test model

        yptr2 = exp.predict(dtr2, return_last=False)#; yptr2=np.exp(yptr2)
        yptr_z2 = exp.predict(dtr2, return_last=False, return_ordinal_z=True); yptr_z2 = yptr_z2[:,:,0]
        yptr2_avg = exp.model.output_layer.get_proba([yptr_z2[ii,20:dtr2.lengths[ii]].mean() for ii in range(len(yptr2))])
        ypva2 = exp.predict(dva2, return_last=False)#; ypva2=np.exp(ypva2)
        ypva_z2 = exp.predict(dva2, return_last=False, return_ordinal_z=True); ypva_z2 = ypva_z2[:,:,0]
        ypva2_avg = exp.model.output_layer.get_proba([ypva_z2[ii,20:dva2.lengths[ii]].mean() for ii in range(len(ypva2))])
        ypte2, Hte2 = exp.predict(dte2, output_id=[0,1], return_last=False)#; ypte2=np.exp(ypte2)
        ypte_z2 = exp.predict(dte2, return_last=False, return_ordinal_z=True); ypte_z2 = ypte_z2[:,:,0]
        ypte2_avg = exp.model.output_layer.get_proba([ypte_z2[ii,20:dte2.lengths[ii]].mean() for ii in range(len(ypte2))])
        report_performance(dte2.y, np.argmax(ypte2_avg, axis=1), prefix='recurrent te2')

        if tosave:
            with open(result_path, 'wb') as f:
                pickle.dump({
                        'tr':dict(dtr2.drop_X()), 'yptr':yptr2, 'yptr_avg':yptr2_avg,
                        'va':dict(dva2.drop_X()), 'ypva':ypva2, 'ypva_avg':ypva2_avg,
                        'te':dict(dte2.drop_X()), 'ypte':ypte2, 'ypte_avg':ypte2_avg, 'Hte':Hte2
                    }, f, protocol=2)

