import pickle
import sys
import numpy as np
from scipy.signal import savgol_filter
import torch as th
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 10})
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')
sys.path.insert(0, r'../CAMICU_prediction/myml_lib')
from mymodels.eegnet_feedforward import EEGNet_CNN_CAMICU
sys.path.insert(0, r'../RASS_prediction/myml_lib')
from mymodels.eegnet_feedforward import EEGNet_CNN as EEGNet_CNN_RASS
from mymodels.eegnet_recurrent import EEGNet_RNN, ModuleSequence



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
        
    with open('grad.pickle', 'rb') as ff:
        res = pickle.load(ff)
    dataset = res['dataset']
    yp = res['yp']
    ypz = res['ypz']
    X = res['X']
    grad = res['grad']
    grad_env = res['grad_env']

    Fs = 62.5
    channels = ['Fp1-F7', 'Fp2-F8']
    nch = len(channels)
    RASS_levels = [-5,-4,-3,-2,-1,0]
    grad_thres = 90

    tostudy_rass = 0
    #tostudy_rass = -5

    if tostudy_rass==-5:
        tostudy_patient = 'icused11'
        tostudyid = 69
        zoom_start = int(Fs*60*1.37)
    else:
        tostudy_patient = 'icused169'
        tostudyid = 1530
        zoom_start = int(Fs*60*1.37)
    print(tostudy_patient)
    print(tostudy_rass)
    print(tostudyid)

    with open('../RASS_prediction/RASS_folds_info_10-fold.pickle','rb') as ff:
        foldnames, tr_label_timess, va_label_timess, te_label_timess = pickle.load(ff)
    tostudy_fold = [foldnames[i] for i in range(len(te_label_timess)) if tostudy_patient in set([x.split('\t')[0] for x in te_label_timess[i]])][0]
    zoom_T = int(Fs*30)
    grad_offset = 55
    cmap = matplotlib.cm.get_cmap('jet')
    colors = [cmap(1-1./5*i) for i in range(len(RASS_levels))]
    panel_xoffset = -0.06
    panel_yoffset = 1.05

    model_cnn = EEGNet_CNN_RASS('ordinal', 6)
    model_cnn.load_state_dict(th.load('../RASS_prediction/models/model_RASS_cnn_fold 1.pth'))
    model_rnn1 = EEGNet_RNN('ordinal', 192, 6, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=16, rnn_dropout=0.5)
    model_rnn1.load_state_dict(th.load('../RASS_prediction/models/model_RASS_rnn1_%s.pth'%tostudy_fold))
    model_rnn2 = EEGNet_RNN('ordinal', 16, 6, model_type='lstm', rnn_layer_num=1, rnn_hidden_num=4, rnn_dropout=0.2)
    model_rnn2.load_state_dict(th.load('../RASS_prediction/models/model_RASS_rnn2_%s.pth'%tostudy_fold))
    model = ModuleSequence([model_cnn, model_rnn1, model_rnn2])
    model.eval()
    model = model.cuda()
    mus = model.modules[-1].output_layer.get_mus().astype(float)

    label_time = dataset['label_times'][tostudyid]
    y = dataset['y'][tostudyid]
    yp = np.repeat(yp[tostudyid], 250, axis=0)
    ypz = np.repeat(ypz[tostudyid], 250, axis=0)
    X = X[tostudyid]
    grad = grad[tostudyid]
    grad_env = grad_env[tostudyid]
    grad_env = savgol_filter(grad_env, 401, 5)
    grad_thres = np.percentile(grad_env.flatten(), grad_thres)

    L = yp.shape[0]//5*4
    yp = yp[L:]
    ypz = ypz[L:]
    X = X[:,L:]
    grad = grad[:,L:]
    grad_env = grad_env[:,L:]
    grad_scale = 10./np.abs(grad_env).max()

    Xim = np.array(X, copy=True)  # important part
    Xim[grad_env<grad_thres] = np.nan
    Xbg = np.array(X, copy=True)  # unimportant (background) part
    Xbg[grad_env>=grad_thres] = np.nan


    figsize = (10,6.9)
    times = np.arange(Xim.shape[1])/Fs/60.
    plt.close()
    if display_type=='pdf':
        fig=plt.figure(figsize=figsize,dpi=600)
    else:
        fig=plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3,1)

    ax = fig.add_subplot(gs[0,0])
    thresholds = [(mus[i]+mus[i+1])/2 for i in range(len(mus)-1)]
    ylim = [min(-1.6, ypz.min()-0.2), max(1.6, ypz.max()+0.2)]
    ax.plot([times.min(), times.max()], np.array([thresholds,thresholds]), 'k--')
    thresholds = sorted(ylim+thresholds)
    for i in range(len(thresholds)-1):
        ax.fill_between([times.min(), times.max()], [thresholds[i]]*2, [thresholds[i+1]]*2, color=colors[i], edgecolor='none', alpha=0.4)
        ax.text(0.03, (thresholds[i]+thresholds[i+1])/2., 'RASS %d'%(i-5,), ha='left', va='center', color='k')
    ax.plot(times, ypz, c='b', lw=2, clip_on=False)
    if tostudy_rass==-5:
        yy = ylim[1]-0.2
    elif tostudy_rass==0:
        yy = -1.1
    tt = ax.text(1.82, yy, '%s     Label RASS = %d     Predicted Final RASS = %d'%('    '.join(np.array(label_time.split('\t'))[[0,2]]), y-5, np.argmax(yp[-1])-5),ha='right',va='top',color='k')
    tt.set_bbox(dict(facecolor='w', alpha=0.4, edgecolor='none'))
    ax.set_xlim([times.min(), times.max()])
    ax.set_xticklabels([])
    ax.tick_params(axis='x',direction='in')
    ax.set_ylim(ylim)
    ax.set_yticks(thresholds[1:-1])
    ax.set_yticklabels([])
    ax.set_ylabel('Predicted z score')
    seaborn.despine(ax=ax)
    ax.text(panel_xoffset,panel_yoffset,'a',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')

    ax = fig.add_subplot(gs[1,0])
    ch_offset = np.linspace(0, 90*nch, nch)
    ylim = [ch_offset.min()-50, ch_offset.max()+grad_offset+50]
    ax.plot(times, Xim.T+ch_offset, c='r', clip_on=False)
    ax.plot(times, Xbg.T+ch_offset, c='k', clip_on=False)
    ax.plot(times, grad.T*grad_scale+ch_offset+grad_offset, c='k', clip_on=False)
    ax.plot([times[zoom_start], times[zoom_start], times[zoom_start+zoom_T], times[zoom_start+zoom_T], times[zoom_start]], [ylim[0]+20,ylim[1]-20,ylim[1]-20,ylim[0]+20,ylim[0]+20], 'b--',lw=2)
    plt.plot([times[zoom_start], 0],[ylim[0]+20,-105], 'b--', clip_on=False)
    plt.plot([times[zoom_start+zoom_T], times.max()],[ylim[0]+20,-105], 'b--', clip_on=False)
    ax.plot([0.04]*2, [np.mean(ylim)-25, np.mean(ylim)+25], 'k', lw=3)
    #ax.text(0.03,np.mean(ylim),'50uV', ha='left',va='center')
    ax.set_xlim([times.min(), times.max()])
    ax.set_xlabel('Time (min)')
    ax.xaxis.set_label_coords(0.5, -0.084)
    ax.tick_params(axis='x',direction='in')
    ax.set_yticks(np.r_[ch_offset, ch_offset+grad_offset])
    ax.set_yticklabels(channels+['Gradient' for x in channels])
    ax.set_ylim(ylim)
    ax.text(panel_xoffset,panel_yoffset,'b',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax)

    # zoom out
    ax = fig.add_subplot(gs[2,0])
    zoom_times = times[zoom_start:zoom_start+zoom_T]*60
    ax.plot(zoom_times, Xim.T[zoom_start:zoom_start+zoom_T]+ch_offset, c='r', clip_on=False)
    ax.plot(zoom_times, Xbg.T[zoom_start:zoom_start+zoom_T]+ch_offset, c='k', clip_on=False)
    ax.plot(zoom_times, grad.T[zoom_start:zoom_start+zoom_T]*grad_scale+ch_offset+grad_offset, c='k', clip_on=False)
    #ax.plot([22.2]*2, [np.mean(ylim)-25, np.mean(ylim)+25], 'k', lw=3)
    #ax.text(22.44,np.mean(ylim),'50uV', ha='left',va='center')
    ax.set_xlim([zoom_times.min(), zoom_times.max()])
    ax.set_xlabel('Time (s)')
    ax.tick_params(axis='x',direction='in')
    ax.set_yticks(np.r_[ch_offset, ch_offset+grad_offset])
    ax.set_yticklabels(channels+['Gradient' for x in channels])
    ax.set_ylim(ylim)#[0],ylim[1]-70])
    ax.text(panel_xoffset,panel_yoffset,'c',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.176)
    if display_type=='pdf':
        plt.savefig('figure5_grad_%s_RASS%d_%d.pdf'%(tostudy_patient, tostudy_rass, tostudyid),dpi=600,bbox_inches='tight',pad_inches=0.01)
    elif display_type=='png':
        plt.savefig('figure5_grad_%s_RASS%d_%d.png'%(tostudy_patient, tostudy_rass, tostudyid),bbox_inches='tight',pad_inches=0.01)
    else:
        plt.show()
