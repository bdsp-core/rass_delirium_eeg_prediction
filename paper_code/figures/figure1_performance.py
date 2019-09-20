import os
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import *
from scipy.interpolate import InterpolatedUnivariateSpline as interp1d
from scipy.optimize import minimize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score, confusion_matrix, balanced_accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from kw_dunn import kw_dunn
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 9.5})
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')


def report_performance(y, yp, patients, prefix='', Nbt=1000, random_state=None):
    np.random.seed(random_state)
    
    ndim = yp.ndim
    if ndim==2:
        yp2d = yp
        yp = np.argmax(yp, axis=1)
    
    y = y.astype(int)
    cw = compute_class_weight('balanced', [0,1,2,3,4,5], y)
    ww = cw[y]
    
    unique_patients = np.unique(patients)
    diff = np.abs(y-yp)
    maes = np.array([np.sum(diff[patients==upt]*ww[patients==upt])/ww[patients==upt].sum() for upt in unique_patients])
    acc0s = np.array([np.sum((diff[patients==upt]==0)*ww[patients==upt])/ww[patients==upt].sum() for upt in unique_patients])
    acc1s = np.array([np.sum((diff[patients==upt]<=1)*ww[patients==upt])/ww[patients==upt].sum() for upt in unique_patients])
    acc2s = np.array([np.sum((diff[patients==upt]<=2)*ww[patients==upt])/ww[patients==upt].sum() for upt in unique_patients])
    
    kappa6 = cohen_kappa_score(y, yp)
    kappa6_bt = []
    for bti in range(Nbt):
        btids = np.random.choice(len(y), len(y), replace=True)
        kappa6_bt.append(cohen_kappa_score(y[btids], yp[btids]))
        
    mapping = {0:0,1:0,2:1,3:1,4:2,5:2}
    y3 = np.array([mapping[i] for i in y if i in mapping])
    yp3 = np.array([mapping[i] for i in yp if i in mapping])
    kappa3 = cohen_kappa_score(y3, yp3)
    kappa3_bt = []
    for bti in range(Nbt):
        btids = np.random.choice(len(y3), len(y3), replace=True)
        kappa3_bt.append(cohen_kappa_score(y3[btids], yp3[btids]))
    
    print('%s MAE %.3f, acc0 %.3f, acc1 %.3f, acc2 %.3f, kappa6 %.3f, kappa3 %.3f'%(prefix, np.median(maes), np.median(acc0s), np.median(acc1s), np.median(acc2s), kappa6, kappa3))
    #print(pearsonr(y,yp), spearmanr(y,yp))
    return maes, acc0s, acc1s, acc2s, kappa6, kappa6_bt, kappa3, kappa3_bt
    

def bootstrap_ROC(y, yp, Nbt=1000, alpha=0.05, return_interval=True, random_state=None):
    auc = roc_auc_score(y, yp) 
    fpr, tpr, _ = roc_curve(y, yp)
    fpr2 = np.sort(np.unique(fpr))
    
    tprs = []
    aucs = []
    if random_state is not None:
        np.random.seed(random_state)
    for _ in range(Nbt):
        bt_ids = np.random.choice(len(y), len(y), replace=True)
        y_bt = y[bt_ids]
        yp_bt = yp[bt_ids]

        xx, yy, _ = roc_curve(y_bt, yp_bt)
        xx_unique = np.sort(np.unique(xx))
        yy = np.array([yy[xx==xxx].mean() for xxx in xx_unique])
        xx = xx_unique
        foo = interp1d(xx, yy, k=3)#, kind='cubic')
        tprs.append(foo(fpr2))

        aa = roc_auc_score(y_bt, yp_bt)
        aucs.append(aa)
        
    fpr, tpr, tt = roc_curve(y, yp)
    
    tprs = np.array(tprs)
    aucs = np.array(aucs)
    if return_interval:
        auc_CI = np.percentile(aucs, (alpha*100/2., 100-alpha*100/2.))
        tpr_CI = np.percentile(tprs, (alpha*100/2., 100-alpha*100/2.), axis=0)
    else:
        auc_CI = aucs
        tpr_CI = tprs
    return auc, auc_CI, fpr, tpr, tt, fpr2, tpr_CI


def get_best_operation_point(y, yp, method):
    # Unal, I., 2017.
    # Defining an optimal cut-point value in roc analysis: an alternative approach.
    # Computational and mathematical methods in medicine, 2017.
    fpr, tpr, tt = roc_curve(y, yp)
    
    if method.lower().startswith('se') or method.lower().startswith('sp'):
        #fpr_unique = np.sort(np.unique(fpr))
        #tpr_unique = np.array([tpr[fpr==x].mean() for x in fpr_unique])
        #foo = interp1d(tpr_unique, fpr_unique, kind='cubic')
        if method.lower().startswith('se'):
            tpr_best_ = float(method[2:])
            tpr_best = tpr_best_
            #fpr_best = foo(tpr_best)
            best_i = np.where(np.abs(tpr-tpr_best_)<=np.min(np.abs(tpr-tpr_best_)))[0] # possible to have multiple min
            best_i = best_i[np.argmin(fpr[best_i])]
        else:
            fpr_best_ = 1-float(method[2:])
            fpr_best = fpr_best_
            #tpr_best = foo(fpr_best)
            best_i = np.where(np.abs(fpr-fpr_best_)<=np.min(np.abs(fpr-fpr_best_)))[0] # possible to have multiple min
            best_i = best_i[np.argmax(tpr[best_i])]
    elif method=='chi2':
        chi2s = []
        N = len(y)
        ids = []
        for i in range(len(tt)):
            #se = tpr[i]
            #sp = 1-fpr[i]
            s = np.sum((y==0)&(yp<=tt[i]))
            r = np.sum((y==0)&(yp>tt[i]))
            u = np.sum((y==1)&(yp<=tt[i]))
            v = np.sum((y==1)&(yp>tt[i]))
            if s+r==0 or u+v==0 or s+u==0 or r+v==0:
                continue
            chi2 = 1.*N*(s*v-u*r)**2 / (s+r) / (u+v) / (s+u) / (r+v)
            ids.append(i)
            chi2s.append(chi2)
        best_i = np.array(ids)[np.argmax(chi2s)]
    elif method=='Youden':
        best_i = np.argmax(tpr-fpr)
    elif method=='closest01':
        best_i = np.argmin((1-tpr)**2+fpr**2)
    elif method=='CZ':
        best_i = np.argmax(tpr*(1-fpr))
    elif method=='IU':
        auc = roc_auc_score(y, yp)
        best_i = np.argmin(np.abs(tpr-auc) + np.abs(1-fpr-auc))
    else:
        raise NotImplementedError('Unknown method %s'%method)
        
    return fpr[best_i], tpr[best_i], tt[best_i], best_i
        

def pearsonr_CI(x, y, alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''
    r, p = pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi



    
if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)

    random_state = 2019
    types = ['waveform\n(CNN+LSTM)', 'waveform\n(CNN)', 'spect\n(LSTM)', 'bp\n(LSTM)', 'bp+BSR\n(ordinal reg)', 'tech vs.\nnurse']
    key2path = {'waveform\n(CNN+LSTM)':'../RASS_prediction/results',
                'waveform\n(CNN)': '../RASS_prediction/results',
                'spect\n(LSTM)': '../RASS_prediction/results_spect',
                'bp\n(LSTM)': '../RASS_prediction/results_bandpower',
                'bp+BSR\n(ordinal reg)': '../RASS_prediction/results_baselinemodel/results_rass.pickle',
                'tech vs.\nnurse': 'IRR.pickle'}
    ntypes = len(types)
    
    y = {}
    yp = {}
    patients = {}
    maes = {}
    acc0s = {}
    acc1s = {}
    acc2s = {}
    for type_ in types:
        result_path = key2path[type_]

        if type_=='bp+BSR\n(ordinal reg)':
            with open(result_path, 'rb') as ff:
                ys_rass, yps_rass, patients_rass = pickle.load(ff)
            clf = 'ordinal_regression'
            this_y = np.array(ys_rass[clf])
            this_yp = np.array(yps_rass[clf])
            this_patients = np.array(patients_rass[clf])

        elif type_=='tech vs.\nnurse':
    
            # load tech performance

            with open(result_path, 'rb') as ff:
                ynurses, yresearchers, time_diffs, patients_, researchers = pickle.load(ff)
            this_y = np.concatenate([ynurses[r] for r in researchers])
            this_yp = np.concatenate([yresearchers[r] for r in researchers])
            this_patients = np.concatenate([patients_[r] for r in researchers])
            
            ids = np.in1d(this_y, [-5,-4,-3,-2,-1,0])&np.in1d(this_yp, [-5,-4,-3,-2,-1,0])
            this_y = this_y[ids]+5
            this_yp = this_yp[ids]+5
            this_patients = this_patients[ids]
        else:
            result_folds = sorted(filter(lambda x:x.startswith('results_RASS_fold') and x.endswith('.pickle'), os.listdir(result_path)), key=lambda x:int(x[17:-7]))
            this_y = []
            this_yp = []
            this_yp_cnn = []
            this_patients = []
            for result_file in result_folds:
                with open(os.path.join(result_path, result_file), 'rb') as ff:
                    res = pickle.load(ff)
                this_patients.extend(res['te']['patients'])
                this_y.extend(res['te']['y'])
                if type_=='waveform\n(CNN)':
                    this_yp_cnn.extend(res['ypte_avg_cnn'])
                else:
                    this_yp.extend(res['ypte_avg'])
            
        patients[type_] = np.array(this_patients)
        y[type_] = np.array(this_y)
        if type_=='waveform\n(CNN)':
            yp[type_] = np.array(this_yp_cnn)
        else:
            yp[type_] = np.array(this_yp)
        maes[type_], acc0s[type_], acc1s[type_], acc2s[type_], _,_,_,_ = report_performance(y[type_], yp[type_], patients[type_], prefix=type_, Nbt=0, random_state=random_state)
        
        
    panel_xoffset = -0.11
    panel_yoffset = 1.01
    colors = 'krgbm'
    cmap = cm.get_cmap('Set3')
    colors2 = ['lightgray', 'pink', 'lightgreen', 'lightblue'] + [cmap(0.8), cmap(0.1)]
    medianprops = dict(linewidth=2, color='k')
    figsize = (8.8, 6.4)
    plt.close()
    if display_type=='pdf':
        fig=plt.figure(figsize=figsize,dpi=600)
    else:
        fig=plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2,2, width_ratios=[3,2])

    # RASS MAE
    ax = fig.add_subplot(gs[0,0])
    bp = ax.boxplot([maes[type_] for type_ in types], labels=types, medianprops=medianprops, patch_artist=True)
    for patch, cc in zip(bp['boxes'], colors2):
        patch.set_facecolor(cc)
    H, p_omnibus, Z_pairs, p_corrected, reject = kw_dunn([maes[type_] for type_ in types])
    # plot significance stars
    xoffset = 0.2
    yoffset = 0.32
    cc = 0
    cc2 = 0
    yorder = np.arange(100)
    x1s = []
    x2s = []
    ys = []
    print('\nRASS MAE')
    print(p_corrected)
    for i in range(ntypes):
        for j in range(i+1,ntypes):
            if reject[cc]:
                print('%s %s %s (p = %f)'%(types[i].replace('\n',' '), '>' if np.median(maes[types[i]])>np.median(maes[types[j]]) else '<', types[j].replace('\n',' '), p_corrected[cc]))
                x1s.append(i+1)
                x2s.append(j+1)
                ys.append(yorder[cc2])
                cc2 += 1
            cc += 1
    ys[-1] = ys[2]
    for i, j, yo in zip(x1s, x2s, ys):
        ax.plot([i+xoffset]*2+[j-xoffset]*2, np.array([3.2,3.3,3.3,3.2])+yo*yoffset+yoffset, c='k')
        #ax.text((i+j)/2., 3.3+yo*yoffset+yoffset, '*', ha='center', va='bottom', weight='bold')#fontsize=16)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_ylabel('RASS: Mean absolute error')
    ax.yaxis.grid(True)
    ax.text(panel_xoffset,panel_yoffset,'a',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax)

    # RASS AUC
    
    auc_CIs = {}
    for i, type_ in enumerate(types[:-1]):
        ids = np.in1d(y[type_], [0,1,4,5])
        yy = (y[type_][ids]>2).astype(float)
        #yyp = yp[type_][ids][:,-2:].sum(axis=1)
        yyp = yp[type_][ids][:,-2:].sum(axis=1)/(yp[type_][ids][:,:2].sum(axis=1)+yp[type_][ids][:,-2:].sum(axis=1))
        auc, auc_CI, fpr, tpr, tt, fpr_CI, tpr_CI = bootstrap_ROC(yy, yyp, return_interval=False, random_state=random_state+i)
        auc_CIs[type_] = auc_CI
    H, p_omnibus, Z_pairs, p_corrected, reject = kw_dunn([auc_CIs[type_] for type_ in types[:-1]])
    cc = 0
    print('\nRASS AUC')
    print(p_corrected)
    for i in range(ntypes-1):
        for j in range(i+1,ntypes-1):
            if reject[cc]:
                print('%s %s %s (p = %f)'%(types[i].replace('\n',' '), '>' if np.median(auc_CIs[types[i]])>np.median(auc_CIs[types[j]]) else '<', types[j].replace('\n',' '), p_corrected[cc]))
            cc += 1
            
    ticks = np.arange(0,1.1,0.1)
    ticklabels = ['0','','0.2','','0.4','','0.6','','0.8','','1']
    random_state = 2018
    ax = fig.add_subplot(gs[0,1])
    ax.plot([0,1], [0,1], 'k--')
    for i, type_ in enumerate(types):
        if type_=='tech vs.\nnurse':
            continue
        ids = np.in1d(y[type_], [0,1,4,5])
        yy = (y[type_][ids]>2).astype(float)
        #yyp = yp[type_][ids][:,-2:].sum(axis=1)
        yyp = yp[type_][ids][:,-2:].sum(axis=1)/(yp[type_][ids][:,:2].sum(axis=1)+yp[type_][ids][:,-2:].sum(axis=1))
        fpr, tpr, _ = roc_curve(yy, yyp)
        auc = roc_auc_score(yy, yyp)
        ax.plot(fpr, tpr, c=colors[i], lw=2, alpha=0.8, label='%s: %.2f'%(type_.replace('\n',' '), auc))
        #print(type_, auc, auc_CI)
    ax.legend(loc='lower right', facecolor='w', edgecolor='none', frameon=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('RASS -5,-4 vs -1,0: 1 - specificity')
    ax.set_ylabel('RASS -5,-4 vs -1,0: sensitivity')
    ax.text(panel_xoffset,panel_yoffset,'b',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    seaborn.despine(ax=ax)

    # RASS acc
    tols = [0,1]#,2]
    ax = fig.add_subplot(gs[1,0])
    xpos = np.concatenate([np.arange(ntypes)+(ntypes+1)*tt for tt in tols])
    accs = []
    for tt in tols:
        if tt==0:
            accs_ = acc0s
        elif tt==1:
            accs_ = acc1s
        elif tt==2:
            accs_ = acc2s
        H, p_omnibus, Z_pairs, p_corrected, reject = kw_dunn([accs_[type_] for type_ in types])
        cc = 0
        print('\nRASS Acc%d'%tt)
        print(p_corrected)
        for i in range(ntypes):
            for j in range(i+1,ntypes):
                if reject[cc]:
                    print('%s %s %s (p = %f)'%(types[i].replace('\n',' '), '>' if np.median(accs_[types[i]])>np.median(accs_[types[j]]) else '<', types[j].replace('\n',' '), p_corrected[cc]))
                cc += 1
        for type_ in types:
            accs.append(accs_[type_]*100.)
    print([np.median(acc) for acc in accs])
    bp = ax.boxplot(accs, positions=xpos, medianprops=medianprops, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors2[i%ntypes])
    xticks = np.concatenate([[(ntypes-1)/2.+(ntypes+1)*tt] for tt in tols])
    xticklabels = ['y=y\'' if tt==0 else '|y-y\'| <= %d'%tt for tt in tols]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_ylabel('RASS: Accuracy (%)')
    ax.yaxis.grid(True)
    ax.set_ylim([-5,105])
    ax.text(panel_xoffset,panel_yoffset,'c',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax)

    # CAM-ICU AUC
    ax = fig.add_subplot(gs[1,1])
    ax.plot([0,1], [0,1], 'k--')

    types = ['CNN+LSTM', 'logistic_regression', 'svm', 'rf']
    ntypes = len(types)
    types_txt = ['waveform (CNN+LSTM)', 'bp+BSR (log reg)', 'bp+BSR (SVM)', 'bp+BSR (RF)']
    with open('../RASS_prediction/results_baselinemodel/results_camicu.pickle', 'rb') as ff:
        res_baseline = pickle.load(ff)
                
    n_fix_layer = 5
    
    auc_CIs = []
    for i, type_ in enumerate(types):
        if type_=='CNN+LSTM':
            y = []
            yp = []
            for jj in range(10):
                file_path = '../CAMICU_prediction/results/results_CAMICU_fold %d_nfix%d.pickle'%(jj+1,n_fix_layer)
                with open(file_path, 'rb') as ff:
                    res = pickle.load(ff)
                y.extend(res['te']['y'][:,-1])
                yp.extend(res['ypte_avg'])
            y = np.array(y)
            yp = np.array(yp)

            auc, auc_CI, fpr, tpr, tt, fpr_CI, tpr_CI = bootstrap_ROC(y, yp, return_interval=False, random_state=random_state+i)
            auc_CIs.append(auc_CI)
            sens = []
            spes = []
            thres = []
            methods = ['se0.6', 'se0.7', 'se0.8', 'se0.9',
                       'sp0.6', 'sp0.7', 'sp0.8', 'sp0.9',
                       'chi2', 'Youden', 'CZ']#, 'IU', 'closest01'
            for method in methods:
                fpr_best, tpr_best, thres_best, best_i = get_best_operation_point(y, yp, method)
                sens.append(tpr_best)
                spes.append(1-fpr_best)
                thres.append(thres_best)
            perf_table = pd.DataFrame(data={'tols': thres, 'Sensitivity':sens, 'Specificity': spes}, index=methods)
            perf_table = perf_table[['tols', 'Sensitivity', 'Specificity']]
            perf_table.to_csv('delirium_operation_points.csv', sep=',', index=True)
            print(perf_table)

            markers = 'o*s^v'
            ids = list(range(len(perf_table)-1, len(perf_table)))
            #ax.fill_between(fpr_CI, tpr_CI[0], tpr_CI[1], color=colors[i], alpha=0.3)
            for i in range(len(ids)):
                ax.text(1-perf_table.loc[methods[ids[i]], 'Specificity'], perf_table.loc[methods[ids[i]], 'Sensitivity']+0.05,
                        'optimal\npoint', ha='center', va='bottom')
                ax.scatter([1-perf_table.loc[methods[ids[i]], 'Specificity']], [perf_table.loc[methods[ids[i]], 'Sensitivity']],
                           c='r', s=70, marker=markers[i])#, label='optimal point')#methods[ids[i]])
            ax.plot(fpr, tpr, lw=2, c=colors[i], label='%s: %.2f'%(types_txt[i], auc))#, auc_CI[0], auc_CI[1]))
        else:
            y = res_baseline[0][type_]
            yp = res_baseline[1][type_]
            auc, auc_CI, fpr, tpr, tt, fpr_CI, tpr_CI = bootstrap_ROC(y, yp, return_interval=False, random_state=random_state+i)
            auc_CIs.append(auc_CI)
            ax.plot(fpr, tpr, lw=2, c=colors[i], label='%s: %.2f'%(types_txt[i], auc))#, auc_CI[0], auc_CI[1]))
            
    print('\nCAM-ICU AUC')
    H, p_omnibus, Z_pairs, p_corrected, reject = kw_dunn(auc_CIs)
    print(p_corrected)
    cc = 0
    for i in range(ntypes):
        for j in range(i+1,ntypes):
            if reject[cc]:
                print('%s %s %s (p = %f)'%(types[i].replace('\n',' '), '>' if np.median(auc_CIs[i])>np.median(auc_CIs[j]) else '<', types[j].replace('\n',' '), p_corrected[cc]))
            cc += 1
            
    ax.legend(loc='lower right', facecolor='w', edgecolor='none', frameon=True, framealpha=0.8)
    ax.set_xlabel('CAM-ICU: 1 - specificity')
    ax.set_ylabel('CAM-ICU: sensitivity')
    ax.grid(True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.text(panel_xoffset,panel_yoffset,'d',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    seaborn.despine(ax=ax)

    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig('figure1_performance.pdf',dpi=600,bbox_inches='tight',pad_inches=0.01)
    elif display_type=='png':
        plt.savefig('figure1_performance.png',bbox_inches='tight',pad_inches=0.01)
    else:
        plt.show()

