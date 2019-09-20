import pickle
import numpy as np
from scipy.signal import hilbert
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from mord import *
from mne.time_frequency import psd_array_multitaper
from tqdm import tqdm
import sys
sys.path.insert(0, r'myml_lib')
from dataset import *


def generate_features(dataset, Fs):
    batchids = np.array_split(dataset.select_ids, 100)
    
    features = []
    for batchid in tqdm(batchids):
        spec, freq = psd_array_multitaper(dataset.X[batchid], Fs, fmin=0.5, fmax=20,
                                          bandwidth=1., adaptive=False, low_bias=True,
                                          normalization='full', n_jobs=-1, verbose=False)
        df = freq[1]-freq[0]
        spec = spec.mean(axis=2)
        
        delta_bp = spec[:,:,freq<=4].sum(axis=-1)*df
        theta_bp = spec[:,:,(freq>4)&(freq<=8)].sum(axis=-1)*df
        alpha_bp = spec[:,:,(freq>8)&(freq<=12)].sum(axis=-1)*df
        
        totalpower = spec.sum(axis=-1)
        delta_bp_n = delta_bp/(totalpower*df)
        theta_bp_n = theta_bp/(totalpower*df)
        alpha_bp_n = alpha_bp/(totalpower*df)
        
        X2 = dataset.X[batchid].transpose(0,2,1,3).reshape(len(batchid), dataset.X.shape[2], -1)
        env = np.abs(hilbert(X2)).mean(axis=1)
        
        features.extend(np.c_[delta_bp.mean(axis=1), theta_bp.mean(axis=1), alpha_bp.mean(axis=1),
                              delta_bp_n.mean(axis=1), theta_bp_n.mean(axis=1), alpha_bp_n.mean(axis=1),
                              (env<5).mean(axis=1)])
    
    feature_names = ['delta_power', 'theta_power', 'alpha_power',
                     'delta_power_normalized', 'theta_power_normalized', 'alpha_power_normalized',
                     'burst suppression ratio']
    return np.array(features), feature_names
    

if __name__=='__main__':

    label_type = 'camicu'
    clfs = ['logistic_regression', 'svm', 'rf']
    
    Fs = 62.5  # 62.5Hz since 4x down-sampled from 250Hz
    random_state = 10
    label_mapping = {-5:0,-4:1,-3:2,-2:3,-1:4,0:5}
    
    ## read all data
    data_path_rnn = '/home/sunhaoqi/eeg_segs_%s_recurrent_w4s2_L9.5min.h5'%label_type
    dall_rnn = MyDataset(data_path_rnn, label_type, data_type='eeg', label_mapping=label_mapping, class_weighted=True)
    
    # remove bad quality patients
    bad_quality_patients = ['icused14', 'icused29', 'icused44', 'icused52', 'icused69', 'icused98', 'icused122', 'icused125', 'icused185', 'icused199']
    print('%d patients removed due to bad signal quality'%len(bad_quality_patients))
    select_mark = ~np.in1d(dall_rnn.patients, bad_quality_patients)
    dall_rnn = slice_dataset(dall_rnn, np.where(select_mark)[0])
    
    K = dall_rnn.K
    dall_rnn.summary(suffix='recurrent all')

    ## generate tr, va, te folds
    
    np.random.seed(random_state+10)
    with open('RASS_folds_info_10-fold.pickle','rb') as ff:
        foldnames, tr_label_timess, va_label_timess, te_label_timess = pickle.load(ff)

    feature_path = 'step4_features_%s.pickle'%label_type
    if os.path.exists(feature_path):
        with open(feature_path, 'rb') as ff:
            X, feature_names = pickle.load(ff)
    else:
        X, feature_names = generate_features(dall_rnn, Fs)
        with open(feature_path, 'wb') as ff:
            pickle.dump([X, feature_names], ff, protocol=2)
    delattr(dall_rnn, 'X')
    
    ## train
    ytes = {clf:[] for clf in clfs}
    yptes = {clf:[] for clf in clfs}
    patienttes = {clf:[] for clf in clfs}
    for clfname in clfs:
        for fi, fold, tr_label_times, va_label_times, te_label_times in zip(range(len(foldnames)), foldnames, tr_label_timess, va_label_timess, te_label_timess):
            print('\n########## [%s %d/%d] %s ##########\n'%(clfname, fi+1, len(foldnames), fold))
            
            # generate training and testing sets
            
            te_patients = np.unique([x.split('\t')[0] for x in te_label_times])
            trids = np.where(~np.in1d(dall_rnn.patients, te_patients))[0]
            teids = np.where(np.in1d(dall_rnn.patients, te_patients))[0]
            dtr = slice_dataset(dall_rnn, trids)
            dte = slice_dataset(dall_rnn, teids)

            #dtr.summary(suffix='tr')
            #dte.summary(suffix='te')
            Xtr = X[trids]
            Xte = X[teids]
            ytr = dtr.y
            yte = dte.y
            if label_type == 'camicu':
                ytr = ytr[:,-1]
                yte = yte[:,-1]
            patients_tr = dtr.patients
            patients_te = dte.patients
            
            # remove inf and nan
            
            Xtr[np.isinf(Xtr)] = np.nan
            notnanids = np.where(~np.any(np.isnan(Xtr), axis=1))[0]
            Xtr = Xtr[notnanids]
            ytr = ytr[notnanids]
            patients_tr = patients_tr[notnanids]
            Xte[np.isinf(Xte)] = np.nan
            notnanids = np.where(~np.any(np.isnan(Xte), axis=1))[0]
            Xte = Xte[notnanids]
            yte = yte[notnanids]
            patients_te = patients_te[notnanids]
            label_times_te = dte.label_times[notnanids]
            
            # standardize features
            
            ss = StandardScaler().fit(Xtr)
            Xtr = ss.transform(Xtr)
            Xte = ss.transform(Xte)
            
            # fit model
            
            if clfname=='ordinal_regression':
                clf = LogisticAT()
                params = {'alpha': [0.001,0.01,0.1,1.]}
                
            elif clfname=='logistic_regression':
                clf = LogisticRegression(penalty='l2', class_weight='balanced', random_state=random_state+fi, max_iter=1000, solver='lbfgs')
                params = {'C': [0.01, 0.1,1.,10.,100.]}
                
            elif clfname=='svm':
                clf = SVC(kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=random_state+fi)
                #clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, class_weight='balanced', random_state=random_state+fi)
                params = {'C': [0.01, 0.1,1.,10.,100.]}
                
            elif clfname=='rf':
                clf = RandomForestClassifier(criterion='gini', n_jobs=1, random_state=random_state+fi, class_weight='balanced_subsample')
                params = {'n_estimators':[10,20,50,100]}
            #params.update({'fs__estimator__C':[0.01, 0.1, 1.]})
                
            if clfname=='ordinal_regression':
                resampler = RandomOverSampler(sampling_strategy='auto', return_indices=False, random_state=random_state+fi+1)
                Xtr, ytr = resampler.fit_sample(Xtr, ytr)
                patients_tr = patients_tr[resampler.sample_indices_]
            
            #feature_selector = SelectFromModel(LinearSVC(penalty='l1', class_weight='balanced', random_state=random_state+fi, max_iter=10000, tol=0.01, dual=False))
            cv = GroupKFold(n_splits=10).split(Xtr, ytr, groups=patients_tr)
            #clf = Pipeline([('fs', feature_selector), ('clf', clf)])
            clf = GridSearchCV(clf, params, scoring='balanced_accuracy', n_jobs=10, iid=True, refit=True, cv=cv)#, verbose=10)
            clf.fit(Xtr, ytr)
            print(clf.best_params_)
                
            unique_label_times = np.unique(label_times_te)
            yte = np.array([yte[label_times_te==ul].mean() for ul in unique_label_times])
            patientte = [x.split('\t')[0] for x in unique_label_times]
            
            if label_type=='rass':
                #ypte = clf.predict_proba(Xte)
                zpte = np.asarray(Xte.dot(clf.best_estimator_.coef_), dtype=np.float64)
                zpte = np.array([zpte[label_times_te==ul].mean(axis=0) for ul in unique_label_times])
                ypte = np.c_[np.zeros(len(zpte)), norm.cdf(clf.best_estimator_.theta_[:,None]-zpte).T, np.ones(len(zpte))]
                ypte = np.diff(ypte, axis=1)
            else:
                ypte = clf.predict_proba(Xte)[:,1]
                ypte = np.array([ypte[label_times_te==ul].mean(axis=0) for ul in unique_label_times])
            
            ytes[clfname].extend(yte)
            yptes[clfname].extend(ypte)
            patienttes[clfname].extend(patientte)
        ytes[clfname] = np.array(ytes[clfname])
        yptes[clfname] = np.array(yptes[clfname])
        patienttes[clfname] = np.array(patienttes[clfname])
        
        with open('results_baselinemodel/results_%s.pickle'%label_type, 'wb') as ff:
            pickle.dump([ytes, yptes, patienttes], ff, protocol=2)
        
