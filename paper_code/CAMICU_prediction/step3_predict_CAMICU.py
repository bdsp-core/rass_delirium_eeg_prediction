import pickle
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import torch as th
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
import sys
sys.path.insert(0, r'myml_lib')
from dataset import *
from experiment import *
from mymodels.eegnet_feedforward import EEGNet_CNN, EEGNet_CNN_CAMICU
from mymodels.eegnet_recurrent import EEGNet_RNN, ModuleSequence


tosave = True
n_gpu = 1
label_type = 'camicu'
cv_method = '10-fold'
random_state = 10
batch_size = 16
lr = 0.001
max_epoch = 10
data_type = 'eeg'
CAMICU_window_length = 48  # [s]
CAMICU_window_step = 24    # [s]

if label_type == 'rass':
    label_mapping = {-5:0,-4:1,-3:2,-2:3,-1:4,0:5}
    loss_function = 'ordinal'
elif label_type == 'camicu':
    label_mapping = None
    loss_function = 'bin'
else:
    raise ValueError('Unknown label type %s'%label_type)


def get_nonpositive_rass_ids(dataset, return_mask=False):
    """
    This is not equal to (dall_rnn.y[:,0]<=0).
    This will return ids using label assessment as smallest unit,
    i.e. when some of the labels in an assessment is positive, the entire assessment is excluded.
    """
    mask = np.ones(len(dataset))==1
    for lt in dataset.unique_label_times:
        ids = np.where(dataset.label_times==lt)[0]
        if (lt=='icused152\tCAMICU\t2015-09-29 03:55:00.0\tEmily' or # wrong time, should be 15:55:00
           lt=='icused228\tCAMICU\t2016-05-22 17:22:00.0\tDavid' or # can't find in RASS...
           lt=='icused156\tCAMICU\t2015-10-15 17:30:00.0\tEmily' or # can't find in RASS...
           np.any(dataset.y[ids,0]>0)):
            mask[ids] = False
    if return_mask:
        return mask
    else:
        return np.where(mask)[0]


def adapt_to_CAMICU_folds(dataset, tr_label_timess, va_label_timess, te_label_timess):
    """
    Keep the patients in tr+va/te split the same.
    Randomly split label times in tr+va.
    """
    assert cv_method == '10-fold'
    unique_label_times = np.unique(dataset.label_times)
    tr_label_timess2 = []
    va_label_timess2 = []
    te_label_timess2 = []
    for fi in range(len(tr_label_timess)):
        te_patients = np.unique([x.split('\t')[0] for x in te_label_timess[fi]])
        trva_patients = np.unique([x.split('\t')[0] for x in np.r_[tr_label_timess[fi], va_label_timess[fi]]])
        
        te_lt = np.unique(dataset.label_times[np.in1d(dataset.patients, te_patients)])
        trva_lt = np.unique(dataset.label_times[np.in1d(dataset.patients, trva_patients)])
        np.random.seed(random_state+fi)
        np.random.shuffle(trva_lt)
        tr_lt = trva_lt[:len(trva_lt)//10*9]
        va_lt = trva_lt[len(trva_lt)//10*9:]
        
        tr_label_timess2.append(tr_lt)
        va_label_timess2.append(va_lt)
        te_label_timess2.append(te_lt)
    return tr_label_timess2, va_label_timess2, te_label_timess2


def augment_control_examples(dataset, aug_dataset, aug_result_path):#, random_state=None):
    """
    Add certainly CAMICU=0 examples (RASS==0 and RASS_pred>=-1)
    """
    # step 1: find CAMICU=0 examples (negative control examples) from aug result
    with open(aug_result_path, 'rb') as ff:
        res = pickle.load(ff)
    aug_patients = np.r_[res['tr']['patients'], res['va']['patients']]
    aug_label_times = np.r_[res['tr']['label_times'], res['va']['label_times']]
    aug_seg_start_pos = np.r_[res['tr']['seg_start_pos'], res['va']['seg_start_pos']]
    aug_y = np.r_[res['tr']['y'], res['va']['y']]
    aug_yp = np.argmax(np.r_[res['yptr_avg'], res['ypva_avg']], axis=1)
    #aug_yp_z = np.r_[res['yptr_z'], res['ypva_z']]
    #if aug_yp.ndim==2:
    #    aug_yp = aug_yp[:,-1]
    #if aug_yp_z.ndim==2:
    #    aug_yp_z = aug_yp_z[:,-1]
    
    control_ids = np.where((aug_y==5)&(aug_yp>=4)&np.in1d(aug_patients, dataset.unique_patients))[0]
    #thres = np.percentile(aug_yp_z[control_ids], 40)
    #control_ids = control_ids[aug_yp_z[control_ids]>=thres]
    
    """
    # step 2: if len(control_ids)>len(control_needed), remove some of control_ids
    assert np.sum(dataset.y[:,-1]==1)*1./np.sum(dataset.y[:,-1]==0)>=2  # only augment when #pos/#neg>=2
    n_control_needed = np.sum(dataset.y[:,-1]==1)-np.sum(dataset.y[:,-1]==0)
    
    if len(control_ids)>n_control_needed:
        # remove, but the smallest unit is by assessment
        raise NotImplementedError('len(control_ids)>n_control_needed')
        #unique_assess = np.unique(aug_label_times[control_ids])
        #assess_start_end = np.array([np.percentile(np.where(aug_label_times==ua)[0], (0,100)) for ua in unique_assess])
        #assess_len = assess_start_end[:,1]-assess_start_end[:,0]+1
        #np.random.seed(random_state)
        #np.random.shuffle(assess_len)
        #control_unique_assess = unique_assess[:np.argmin(np.abs(np.cumsum(assess_len)-n_control_needed))+1]
        #control_ids = control_ids[np.in1d(aug_label_times[control_ids], control_unique_assess)]
    elif len(control_ids)<n_control_needed/10.:
        # available controls is much smaller than needed, reduce the positive in the original dataset
        _, idx = np.unique(dataset.label_times, return_index=True)
        idx = np.sort(idx)
        pos_label_times = dataset.unique_label_times[dataset.y[idx,-1]==1]
        neg_label_times = dataset.unique_label_times[dataset.y[idx,-1]==0]
        ids = sum([np.where(dataset.label_times==lt)[0][::2].tolist() for lt in pos_label_times], [])
        ids.extend(np.where(np.in1d(dataset.label_times, neg_label_times))[0])
        ids = sorted(ids)
        dataset = slice_dataset(dataset, ids)
    """
    #control_patients = aug_patients[control_ids]
    control_label_times = aug_label_times[control_ids]
    #control_seg_start_pos = aug_seg_start_pos[control_ids]
    
    # step 3: find the ids the control examples in aug_dataset
    
    # first get keys from controls, key = (label_times+seg_start_pos)
    #control_keys = np.array(['%s\t%s'%(control_label_times[i], control_seg_start_pos[i]) for i in range(len(control_patients))])
    # then get keys from aug_dataset
    with h5py.File(aug_dataset.input_path, 'r') as ff:
        aug_patients = ff['patient'][:]
        aug_label_times = ff['label_times'][:]
        aug_seg_start_pos = ff['seg_start_pos'][:]
        aug_drugs = ff['drug'][:]
        if aug_seg_start_pos.ndim==2:
            aug_seg_start_pos2 = aug_seg_start_pos[:,0]
        else:
            aug_seg_start_pos2 = aug_seg_start_pos
    #aug_keys = np.array(['%s\t%s'%(aug_label_times[i], aug_seg_start_pos2[i]) for i in range(len(aug_patients))])
    # then map
    #control_ids_in_aug = np.where(np.in1d(aug_keys, control_keys))[0]
    control_ids_in_aug = np.where(np.in1d(aug_label_times, control_label_times))[0]
    
    # step 4: generate control_dataset
    
    control_dataset = aug_dataset
    control_dataset.patients = aug_patients[control_ids_in_aug]
    control_dataset.label_times = aug_label_times[control_ids_in_aug]
    control_dataset.seg_start_pos = aug_seg_start_pos[control_ids_in_aug]
    control_dataset.drugs = aug_drugs[control_ids_in_aug]
    control_dataset.len = len(control_dataset.patients)
    control_dataset.select_ids = np.arange(len(control_dataset.patients))
    with h5py.File(aug_dataset.input_path, 'r') as ff:
        control_dataset.X = ff['X'][list(control_ids_in_aug)]
    
    # step 5: add to dataset
    
    control_dataset.y = np.zeros((len(control_dataset), dataset.y.shape[1]))  # labels = negative (0)
    newdataset = combine_dataset(dataset, control_dataset)
    newdataset.X = np.r_[dataset.X, control_dataset.X]
    #newdataset.select_ids = np.arange(len(newdataset))
    newdataset.augmented = np.r_[np.zeros(len(newdataset)-len(control_dataset)), np.ones(len(control_dataset))].astype(int)
    newdataset.set_y_related()
    
    return newdataset


def report_performance(y, yp, prefix=''):
    rmse = np.sqrt(np.mean((y-yp)**2))
    mae = np.mean(np.abs(y-yp))
    msg = '%s RMSE = %g\tMAE = %g'%(prefix, rmse, mae)
    if len(np.unique(y))==2:
        auc = roc_auc_score(y, yp)
        msg += '\tAUC = %g'%auc        
    print(msg)


if __name__=='__main__':
    n_fix_layer = int(sys.argv[1])
    
    ## read all data
    RASS_result_path = '../RASS_prediction/results'
    data_path_ff = '/data/delirium/eeg_segs_%s_feedforward_w4s2.h5'%label_type
    data_path_rnn = '/data/delirium/eeg_segs_%s_recurrent_w4s2_L9.5min.h5'%label_type

    dall_ff = MyDataset(data_path_ff, label_type, data_type=data_type, label_mapping=label_mapping, class_weighted=True)
    dall_rnn = MyDataset(data_path_rnn, label_type, data_type=data_type, label_mapping=label_mapping, class_weighted=True)
    dall_ff_rass = MyDataset(data_path_ff.replace('camicu','rass'), 'rass', data_type=data_type, label_mapping=label_mapping, class_weighted=True, really_load=False)
    dall_rnn_rass = MyDataset(data_path_rnn.replace('camicu','rass'), 'rass', data_type=data_type, label_mapping=label_mapping, class_weighted=True, really_load=False)
    
    # remove bad quality patients
    bad_quality_patients = ['icused14', 'icused29', 'icused44', 'icused52', 'icused69', 'icused98', 'icused122', 'icused125', 'icused185', 'icused199']
    print('%d patients removed due to bad signal quality'%len(bad_quality_patients))
    select_mark = (~np.in1d(dall_rnn.patients, bad_quality_patients))&get_nonpositive_rass_ids(dall_rnn, return_mask=True)
    dall_rnn = slice_dataset(dall_rnn, np.where(select_mark)[0])
    select_mark = (~np.in1d(dall_ff.patients, bad_quality_patients))&get_nonpositive_rass_ids(dall_ff, return_mask=True)
    dall_ff = slice_dataset(dall_ff, np.where(select_mark)[0])

    K = dall_rnn.K
    dall_ff.summary(suffix='feedfoward all')
    dall_rnn.summary(suffix='recurrent all')

    ## generate tr, va, te folds
    # use folds from RASS to keep consistant
    with open('../RASS_prediction/RASS_folds_info_%s.pickle'%cv_method, 'rb') as ff:
        foldnames, tr_label_timess, va_label_timess, te_label_timess = pickle.load(ff)
    tr_label_timess, va_label_timess, te_label_timess = adapt_to_CAMICU_folds(dall_rnn, tr_label_timess, va_label_timess, te_label_timess)
    
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

    ## train

    np.random.seed(random_state+10)
    for fi, fold, tr_label_times, va_label_times, te_label_times in zip(range(len(foldnames)), foldnames, tr_label_timess, va_label_timess, te_label_timess):
        print('\n########## [%d/%d] %s ##########\n'%(fi+1, len(foldnames), fold))
        cnn_model_path = 'models/model_CAMICU_cnn_%s_nfix%d.pth'%(fold, n_fix_layer)
        rnn1_model_path = 'models/model_CAMICU_rnn1_%s_nfix%d.pth'%(fold, n_fix_layer)
        rnn2_model_path = 'models/model_CAMICU_rnn2_%s_nfix%d.pth'%(fold, n_fix_layer)
        result_path = 'results/results_CAMICU_%s_nfix%d.pickle'%(fold, n_fix_layer)
        if os.path.exists(result_path):
            continue

        dtr = slice_dataset(dall_rnn, np.where(np.in1d(dall_rnn.label_times, tr_label_times))[0])
        dva = slice_dataset(dall_rnn, np.where(np.in1d(dall_rnn.label_times, va_label_times))[0])
        dte = slice_dataset(dall_rnn, np.where(np.in1d(dall_rnn.label_times, te_label_times))[0])
    
        # augment by adding CAMICU negative example
        dtr = augment_control_examples(dtr, dall_rnn_rass, os.path.join(RASS_result_path, 'results_RASS_%s.pickle'%fold))
        dva = augment_control_examples(dva, dall_rnn_rass, os.path.join(RASS_result_path, 'results_RASS_%s.pickle'%fold))
        
        # step 1: train CNN

        exp = Experiment(model=EEGNet_CNN_CAMICU(32 if n_fix_layer>0 else 8),
                    batch_size=batch_size, max_epoch=max_epoch, lr=lr,
                    loss_function=loss_function, clip_weight=False,
                    n_gpu=n_gpu, verbose=True, random_state=random_state)
        
        dtr_ff = slice_dataset(dall_ff, np.where(np.in1d(dall_ff.label_times, tr_label_times))[0])
        dva_ff = slice_dataset(dall_ff, np.where(np.in1d(dall_ff.label_times, va_label_times))[0])
        dte_ff = slice_dataset(dall_ff, np.where(np.in1d(dall_ff.label_times, te_label_times))[0])
        dva_ff.set_y_related(dtr_ff.class_weight_mapping)
        
        dtr_ff.summary(suffix='feedfoward tr')
        dva_ff.summary(suffix='feedfoward va')
        dte_ff.summary(suffix='feedfoward te')
        
        if n_fix_layer>0:
            # fix the first *n_fix_layer* layers
            state_dict = th.load('../RASS_prediction/models/model_RASS_cnn_fold 1.pth')
            state_dict = {key:value for key, value in state_dict.items() if not key.startswith('output_layer')}
            state_dict.update({key:value for key, value in exp.model.state_dict().items() if key.startswith('output_layer')})
            key_mapping = [['resblock0.bn1.running_mean', 'bn1.running_mean'],
                           ['resblock0.bn1.running_var', 'bn1.running_var'],
                           ['resblock0.bn2.running_mean', 'bn2.running_mean'],
                           ['resblock0.bn2.running_var', 'bn2.running_var'],
                           ['resblock0.conv1.weight', 'conv2_1.weight'],
                           ['resblock0.conv2.weight', 'conv2_2.weight'],]
            for aa, bb in key_mapping:
                state_dict[aa] = state_dict.pop(bb)
            exp.model.load_state_dict(state_dict)
            for ii in range(n_fix_layer):
                if ii==0:
                    keys = ['resblock%d'%ii, 'first_layer']
                elif ii<=6:
                    keys = ['resblock%d'%ii]
                elif ii==7:
                    keys = ['resblock7', 'bn_output', 'relu_output']
                else:
                    raise NotImplementedError('Don\'t know how to fix layer #%d'%ii)
                for pn, param in exp.model.named_parameters():
                    if pn.split('.')[0] in keys:
                        print(pn)
                        param.requires_grad = False
                        
        print('Model CNN parameters: %d'%exp.model.n_param)
        dtr_ff.fliplr_prob=True
        exp.fit(dtr_ff, Dva=dva_ff, init=False, suffix='_%d'%n_fix_layer)
        dtr_ff.fliplr_prob = False
        if tosave:
            exp.save(cnn_model_path)
        #exp.load(cnn_model_path)
        #print('model loaded from %s'%cnn_model_path)
        
        label_times_cnn_tr = dtr_ff.label_times
        label_times_cnn_va = dva_ff.label_times
        label_times_cnn_te = dte_ff.label_times
        yptr_cnn = exp.predict(dtr_ff)
        ypva_cnn = exp.predict(dva_ff)
        ypte_cnn = exp.predict(dte_ff)
        
        # step 2: get CNN outputs
            
        dva.set_y_related(dtr.class_weight_mapping)
        
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
        exp.model = EEGNet_RNN(loss_function, L, K, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=8, rnn_dropout=0.5)
        print('Model RNN parameters: %d'%exp.model.n_param)
        exp.fit(dtr, Dva=dva, return_last=False, suffix='_%d'%n_fix_layer)
        if tosave:
            exp.save(rnn1_model_path)
        #exp.load(rnn1_model_path)
        #print('model loaded from %s'%rnn1_model_path)
        
        dtr.set_X(np.expand_dims(dtr.X, axis=2))
        dva.set_X(np.expand_dims(dva.X, axis=2))
        dte.set_X(np.expand_dims(dte.X, axis=2))
        dtr2 = make_test_dataset(dtr)
        dva2 = make_test_dataset(dva)
        dte2 = make_test_dataset(dte)
        dva2.set_y_related(dtr2.class_weight_mapping)
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
        exp.model = EEGNet_RNN(loss_function, dtr2.X.shape[2], K, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=4, rnn_dropout=0.2)
        print('Model RNN2 parameters: %d'%exp.model.n_param)
        exp.fit(dtr2, Dva=dva2, return_last=False, suffix='_%d'%n_fix_layer)
        if tosave:
            exp.save(rnn2_model_path)
        #exp.load(rnn2_model_path)
        #print('model loaded from %s'%rnn2_model_path)

        # test model

        yptr2 = exp.predict(dtr2, return_last=False)[:,:,0]
        yptr2_avg = np.array([yptr2[kk, 5:dtr2.lengths[kk]].mean() for kk in range(len(yptr2))])
        ypva2 = exp.predict(dva2, return_last=False)[:,:,0]
        ypva2_avg = np.array([ypva2[kk, 5:dva2.lengths[kk]].mean() for kk in range(len(ypva2))])
        ypte2 = exp.predict(dte2, return_last=False); ypte2 = ypte2[:,:,0]#, Hte2
        ypte2_avg = np.array([ypte2[kk, 5:dte2.lengths[kk]].mean() for kk in range(len(ypte2))])
        
        report_performance(dte2.y[:,-1], ypte2_avg, prefix='recurrent te2')
        
        if tosave:
            with open(result_path, 'wb') as f:
                pickle.dump({
                        'label_times_cnn_tr':label_times_cnn_tr, 'label_times_cnn_va':label_times_cnn_va, 'label_times_cnn_te':label_times_cnn_te,
                        'ytr_cnn': dtr_ff.y, 'yva_cnn': dva_ff.y, 'yte_cnn': dte_ff.y, 
                        'yptr_cnn': yptr_cnn, 'ypva_cnn':ypva_cnn, 'ypte_cnn': ypte_cnn,
                        'tr':dict(dtr2.drop_X()), 'yptr':yptr2, 'yptr_avg':yptr2_avg,
                        'va':dict(dva2.drop_X()), 'ypva':ypva2, 'ypva_avg':ypva2_avg,
                        'te':dict(dte2.drop_X()), 'ypte':ypte2, 'ypte_avg':ypte2_avg
                    }, f, protocol=2)

