import itertools, operator
import pickle
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import detrend
from mne.time_frequency import psd_array_multitaper
import torch as th
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 9})
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')
from braindecode.torch_ext.util import np_to_var, var_to_np
sys.path.insert(0, r'../CAMICU_prediction/myml_lib')
from mymodels.eegnet_feedforward import EEGNet_CNN_CAMICU
sys.path.insert(0, r'../RASS_prediction/myml_lib')
from mymodels.eegnet_feedforward import EEGNet_CNN as EEGNet_CNN_RASS
from mymodels.eegnet_recurrent import EEGNet_RNN, ModuleSequence

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2



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
        

    tostudy_patient = 'icused11'
    with open('../RASS_prediction/RASS_folds_info_10-fold.pickle','rb') as ff:
        foldnames, tr_label_timess, va_label_timess, te_label_timess = pickle.load(ff)
    tostudy_fold = [foldnames[i] for i in range(len(te_label_timess)) if tostudy_patient in set([x.split('\t')[0] for x in te_label_timess[i]])][0]
    print(tostudy_fold)

    res = sio.loadmat('figure3_data_%s.mat'%tostudy_patient)

    start = 26900
    xx = np.array(res['label'][start:,0], copy=True)
    xx[np.isnan(xx)] = -999
    xx = np.diff(xx)
    yy = np.array(res['label'][start:,0], copy=True)[:-1][xx!=0]
    yy = yy[~np.isnan(yy)]
    start = start+np.where(xx!=0)[0][10]

    Fs = 62.5
    T = 24*1
    end = start+int(T*3600/4.)
    y = res['label'][start:end,0].astype(float)
    y_CAMICU = res['label'][start:end,-1].astype(float)
    X = res['EEG'][start:end].astype(float)
    X[np.isnan(X)] = 0.

    model_CAMICU_cnn = EEGNet_CNN_CAMICU(32)
    aa = th.load('../CAMICU_prediction/models/model_CAMICU_cnn_%s_nfix5.pth'%tostudy_fold)
    kk = [x for x in aa.keys() if 'tracked' in x]
    for x in kk: aa.pop(x)
    model_CAMICU_cnn.load_state_dict(aa)

    model_CAMICU_rnn1 = EEGNet_RNN('bin', 192, 2, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=8, rnn_dropout=0.5)
    aa = th.load('../CAMICU_prediction/models/model_CAMICU_rnn1_%s_nfix5.pth'%tostudy_fold)
    kk = [x for x in aa.keys() if 'tracked' in x]
    for x in kk: aa.pop(x)
    model_CAMICU_rnn1.load_state_dict(aa)

    model_CAMICU_rnn2 = EEGNet_RNN('bin', 8, 2, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=4, rnn_dropout=0.2)
    aa = th.load('../CAMICU_prediction/models/model_CAMICU_rnn2_%s_nfix5.pth'%tostudy_fold)
    kk = [x for x in aa.keys() if 'tracked' in x]
    for x in kk: aa.pop(x)
    model_CAMICU_rnn2.load_state_dict(aa)

    model_CAMICU = ModuleSequence([model_CAMICU_cnn, model_CAMICU_rnn1, model_CAMICU_rnn2])
    model_CAMICU.eval()
    model_CAMICU = model_CAMICU.cuda()

    model_RASS_cnn = EEGNet_CNN_RASS('ordinal', 6)
    model_RASS_cnn.load_state_dict(th.load('../RASS_prediction/models/model_RASS_cnn_fold 1.pth'))
    model_RASS_rnn1 = EEGNet_RNN('ordinal', 192, 6, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=16, rnn_dropout=0.5)
    model_RASS_rnn1.load_state_dict(th.load('../RASS_prediction/models/model_RASS_rnn1_%s.pth'%tostudy_fold))
    model_RASS_rnn2 = EEGNet_RNN('ordinal', 16, 6, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=4, rnn_dropout=0.2)
    model_RASS_rnn2.load_state_dict(th.load('../RASS_prediction/models/model_RASS_rnn2_%s.pth'%tostudy_fold))
    model_RASS = ModuleSequence([model_RASS_cnn, model_RASS_rnn1, model_RASS_rnn2])
    model_RASS.eval()
    model_RASS = model_RASS.cuda()
    mus = model_RASS.modules[-1].output_layer.get_mus().astype(float)

    lls = np.arange(0, len(X)+1, 21600)
    ypz = []
    yp_CAMICU = []
    for li in range(len(lls)-1):
        Xinput = np_to_var(np.expand_dims(X[lls[li]:lls[li+1]], 0).astype('float32'))
        Xinput = Xinput.cuda()

        res = model_RASS(Xinput, return_last=False, return_ordinal_z=True)
        ypz.extend(var_to_np(res[0]).flatten().astype(float))
        del res
        th.cuda.empty_cache()
        
        res = model_CAMICU(Xinput, return_last=False)
        yp_CAMICU.extend(var_to_np(res[0]).flatten().astype(float))
        del res
        del Xinput
        th.cuda.empty_cache()
        
    ypz = np.array(ypz)
    yp = model_RASS.modules[-1].output_layer.get_proba(ypz)
    yp_CAMICU = np.array(yp_CAMICU)
    yp2 = np.c_[np.zeros(len(yp)), np.cumsum(yp, axis=1)]
    X = X.transpose(1,0,2).reshape(2,-1)


    figsize = (13,6)
    plt.close()
    if display_type=='pdf':
        fig=plt.figure(figsize=figsize,dpi=600)
    else:
        fig=plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4,1, height_ratios=[5,3,4,3])

    channels = ['Fp1-F7', 'Fp2-F8']
    RASS_levels = [-5,-4,-3,-2,-1,0]
    thresholds = [(mus[i]+mus[i+1])/2 for i in range(len(mus)-1)]
    yticks = np.array([(thresholds[i]+thresholds[i+1])/2 for i in range(len(thresholds)-1)])
    yticks = np.r_[thresholds[0]-(yticks[0]-thresholds[0]), yticks, thresholds[-1]+thresholds[-1]-yticks[-1]]
    times = np.arange(0,X.shape[1],250)/Fs/3600.
    cmap = matplotlib.cm.get_cmap('jet')
    colors = [cmap(1-1./5*i) for i in range(len(RASS_levels))]
    panel_xoffset = -0.035
    panel_yoffset = 1.05
    ax1 = fig.add_subplot(gs[0,0])

    # compute spectrogram
    start_ids = np.arange(0, X.shape[1]-int(Fs*4)+1, int(Fs*2))
    EEG_segs = detrend(X[:,map(lambda x:np.arange(x,x+int(Fs*4)), start_ids)].transpose(1,0,2), axis=2)
    spec, freq = psd_array_multitaper(EEG_segs, Fs, fmin=0, fmax=16, bandwidth=1, adaptive=False, low_bias=True, normalization='full', n_jobs=-1)
    spec[np.isnan(spec)] = 0.
    spec[np.isinf(spec)] = 0.
    spec = 10*np.log10(np.maximum(1e-6,spec))
    spec = spec.mean(axis=1)  # average between channels
    freq = freq[2::2]
    spec = spec[:,2::2]
    print(np.percentile(spec.flatten(),(0,0.1,1,5,10,90,95,99,99.9,100)))
    vmin = -20
    vmax = 20
    ax1.imshow(spec.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='jet', extent=(times.min(), times.max(), freq.min(), freq.max()))
    ax1.set_ylim([freq.min(), freq.max()])
    ax1.set_yticks(np.arange(2,freq.max()+2,2))
    ax1.set_ylabel('Freq (Hz)')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.text(panel_xoffset,panel_yoffset,'a',ha='right',va='top',transform=ax1.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax1)

    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax2.plot(times, y, lw=2, c='k')
    ax2.set_ylim([-5.3,0.3])
    ax2.set_yticks(RASS_levels)
    ax2.set_ylabel('RASS')
    ax2.set_xticks(np.arange(T+1))
    ax2.tick_params(axis='x',direction='in')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.yaxis.grid(True)
    ax2.text(panel_xoffset,panel_yoffset,'b',ha='right',va='top',transform=ax2.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax2)

    max_thres = max(thresholds)
    min_thres = min(thresholds)
    ypz[ypz>max_thres] = np.log1p((ypz[ypz>max_thres]-max_thres)/5.)+max_thres
    #ypz[ypz<min_thres] = min_thres-np.log1p((min_thres-ypz[ypz<min_thres])/5.)

    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
    ax3.plot([times.min(), times.max()], np.array([thresholds,thresholds]), 'k--')
    ax3.plot(times, ypz, c='b', clip_on=False)
    ylim = [-1.2,1.2]#[min(-1.6, ypz.min()-0.1), max(1.6, ypz.max()+0.1)]
    thresholds = sorted(ylim+thresholds)
    for i in range(len(thresholds)-1):
        ax3.fill_between([times.min(), times.max()], [thresholds[i]]*2, [thresholds[i+1]]*2, color=colors[i], edgecolor='none', alpha=0.3)
        tt = ax3.text(0.03, (thresholds[i]+thresholds[i+1])/2., 'RASS %d'%(i-5,), ha='left', va='center', color='k')
        #tt.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='none'))
    ax3.set_ylabel('Predicted z-score')
    ax3.set_ylim(ylim)
    ax3.set_yticks(thresholds[1:-1])
    ax3.set_yticklabels([])
    ax3.set_xticks(np.arange(T+1))
    ax3.tick_params(axis='x',direction='in')
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.text(panel_xoffset,panel_yoffset,'c',ha='right',va='top',transform=ax3.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax3)

    ax4 = fig.add_subplot(gs[3,0], sharex=ax1)
    ax4.plot(times, yp_CAMICU, c='b', label='Predicted probability of CAM-ICU = 1')
    ax4.plot(times, y_CAMICU, c='k', lw=4, label='Nurse assessed CAM-ICU')
    ax4.legend(loc='lower left', ncol=1, facecolor='w', frameon=True, framealpha=0.8)#fancybox=True,
    ax4.set_ylim([-0.05,1.05])
    ax4.set_yticks([0,0.5,1])
    ax4.set_yticklabels(['0','','1'])
    ax4.set_ylabel('CAM-ICU')
    ax4.set_xticks(np.arange(T+1))
    ax4.set_xlabel('time (h)')
    ax4.set_xlim([times.min(), times.max()])
    ax4.tick_params(axis='x',direction='in')
    ax4.yaxis.grid(True)
    ax4.text(panel_xoffset,panel_yoffset,'d',ha='right',va='top',transform=ax4.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax4)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    if display_type=='pdf':
        plt.savefig('figure3_cEEG_example.pdf',dpi=600,bbox_inches='tight',pad_inches=0.01)
    elif display_type=='png':
        plt.savefig('figure3_cEEG_example.png',bbox_inches='tight',pad_inches=0.01)
    else:
        plt.show()
