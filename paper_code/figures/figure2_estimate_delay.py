from itertools import product
import sys
import pickle
import h5py
from tqdm import tqdm
import torch as th
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 11})
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')
from braindecode.torch_ext.util import np_to_var, var_to_np
sys.path.insert(0, r'../RASS_prediction/myml_lib')
from dataset import *
from mymodels.eegnet_feedforward import EEGNet_CNN
from mymodels.eegnet_recurrent import EEGNet_RNN


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
        

    time_offset = 225  # remove the last 225*4/60 = 15min from the first signal, remove the first 15min from the last signal
    data_path = 'figure2_result_offset%d.pickle'%time_offset
    if os.path.exists(data_path):
        with open(data_path, 'rb') as ff:
            all_ypzs, all_yps, all_middle_pos = pickle.load(ff)

    else:
        label_type = 'rass'
        data_type = 'eeg'
        label_mapping = {-5:0,-4:1,-3:2,-2:3,-1:4,0:5}
        dall_rnn = MyDataset('/data/delirium/eeg_segs_%s_recurrent_w4s2_L9.5min.h5'%label_type, label_type, data_type=data_type, label_mapping=label_mapping, class_weighted=True)
        # remove bad quality patients
        bad_quality_patients = ['icused14', 'icused29', 'icused44', 'icused52', 'icused69', 'icused98', 'icused122', 'icused125', 'icused185', 'icused199']
        print('%d patients removed due to bad signal quality'%len(bad_quality_patients))
        select_mark = ~np.in1d(dall_rnn.patients, bad_quality_patients)
        dall_rnn = slice_dataset(dall_rnn, np.where(select_mark)[0])
        dall_rnn = make_test_dataset(dall_rnn)

        folds_path = '../RASS_folds_info_10-fold.pickle'
        with open(folds_path,'rb') as ff:
            foldnames, tr_label_timess, va_label_timess, te_label_timess = pickle.load(ff)
        patient2foldid = {}
        for fi in range(len(foldnames)):
            for pp in te_label_timess[fi]:
                patient2foldid[pp.split('\t')[0]] = fi
        
        models = [th.load('../models/model_RASS_cnnrnn_%s.pth'%fn).cuda() for fn in foldnames]
        all_yps = {}
        all_ypzs = {}
        all_middle_pos = {}
        for label1 in range(-5,1):
            for label2 in range(-5,1):
                if np.abs(label1-label2)<=1:
                    continue
                matched_pos = []
                matched_pts = []
                matched_ypzs = []
                matched_yps = []
                matched_middle_pos = []
                for patient in dall_rnn.unique_patients:
                    patientids = np.where(dall_rnn.patients==patient)[0]
                    pos1 = patientids[dall_rnn.y[patientids]==label1+5]
                    pos2 = patientids[dall_rnn.y[patientids]==label2+5]
                    pos12 = list(product(pos1, pos2))
                    matched_pos.extend(pos12)
                    matched_pts.extend([patient]*len(pos12))
                
                for mpi in tqdm(range(len(matched_pos))):
                    if hasattr(dall_rnn, 'lengths'):
                        length1 = dall_rnn.lengths[matched_pos[mpi][0]]
                        length2 = dall_rnn.lengths[matched_pos[mpi][1]]
                    else:
                        length1 = length2 = dall_rnn.X.shape[1]
                    if length1<=time_offset+20 or length2<=time_offset:
                        continue
                    X = np.concatenate([dall_rnn.X[matched_pos[mpi][0]][:length1-time_offset], dall_rnn.X[matched_pos[mpi][1]][time_offset:length2]], axis=0)
                    length1 -= time_offset
                    length2 -= time_offset

                    Xinput = np_to_var(np.expand_dims(X, 0).astype('float32'))
                    Xinput = Xinput.cuda()

                    model = models[patient2foldid[matched_pts[mpi]]]
                    res = model(Xinput, return_last=False, return_ordinal_z=True)
                    ypz = var_to_np(res[0]).flatten().astype(float)
                    yp = model.modules[-1].output_layer.get_proba(ypz)
                    if np.abs(np.argmax(model.modules[-1].output_layer.get_proba(ypz[20:length1].mean()))-(label1+5))<=0 and \
                                np.abs(np.argmax(model.modules[-1].output_layer.get_proba(ypz[length1:].mean()))-(label2+5))<=0:
                        matched_ypzs.append(ypz)
                        matched_yps.append(yp)
                        matched_middle_pos.append(length1)
                print(label1, label2, len(matched_ypzs))
                all_ypzs[(label1,label2)] = matched_ypzs
                all_yps[(label1,label2)] = matched_yps
                all_middle_pos[(label1, label2)] = matched_middle_pos
                
        with open(data_path, 'wb') as ff:
            pickle.dump([all_ypzs, all_yps, all_middle_pos], ff, protocol=2)


    panel_xoffset = -0.055
    panel_yoffset = 1.01
    figsize = (10,4)
    plt.close()
    if display_type=='pdf':
        fig=plt.figure(figsize=figsize,dpi=600)
    else:
        fig=plt.figure(figsize=figsize)
        
        
    delays = []
    label_diff = []
    for label1 in range(-5,1):
        for label2 in range(-5,1):
            if np.abs(label1-label2)<=1:
                continue
            
            this_delays = []
            for i in range(len(all_yps[(label1,label2)])):
                #part1 = all_yps[(label1,label2)][i][:all_middle_pos[(label1,label2)][i]]
                part2 = all_yps[(label1,label2)][i][all_middle_pos[(label1,label2)][i]:]
                hit = np.where(np.abs(np.argmax(part2, axis=1)-(label2+5))<=1)[0]
                if len(hit)>0:
                    this_delays.append(hit[0]*4)
            delays.extend(this_delays)
            label_diff.extend([label2-label1]*len(this_delays))
            
    delays = np.array(delays)
    label_diff = np.array(label_diff)

    diffs = [2,3,4,5]
    this_delays1 = [delays[np.abs(label_diff)==x] for x in diffs]
    this_delays2 = [delays[label_diff==x] for x in diffs]
    this_delays3 = [delays[label_diff==-x] for x in diffs]
    this_delays1_median = [np.median(x) for x in this_delays1]
    this_delays2_median = [np.median(x) for x in this_delays2]
    this_delays3_median = [np.median(x) for x in this_delays3]
    print(this_delays1_median)
    print(this_delays2_median)
    print(this_delays3_median)

    ylim = [0,1390]
    ax = fig.add_subplot(131)
    #ax.plot(range(1,len(diffs)+1), [np.median(x) for x in this_delays1], c='k')
    bp = ax.boxplot(this_delays1, labels=diffs, widths=0.32)
    #for i, patch in enumerate(bp['boxes']):
    #    patch.set_facecolor(colors2[i%ntypes])
    for i, x in enumerate(this_delays1_median):
        ax.text(1.4+i, x, str(int(round(x))), ha='center', va='center')
    ax.set_xlabel('|RASS2 - RASS1|')
    ax.set_ylabel('Delay (s)')
    ax.yaxis.grid(True)
    ax.set_ylim(ylim)
    ax.text(panel_xoffset-0.08,panel_yoffset,'a',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax)

    ax = fig.add_subplot(132)
    #ax.plot(range(1,len(diffs)+1), [np.median(x) for x in this_delays2], c='k')
    bp = ax.boxplot(this_delays2, labels=diffs, widths=0.32)
    #for i, patch in enumerate(bp['boxes']):
    #    patch.set_facecolor(colors2[i%ntypes])
    for i, x in enumerate(this_delays2_median):
        ax.text(1.4+i, x, str(int(round(x))), ha='center', va='center')
    ax.set_xlabel('RASS2 - RASS1 (increase)')
    ax.yaxis.grid(True)
    ax.set_ylim(ylim)
    ax.set_yticklabels([])
    ax.text(panel_xoffset,panel_yoffset,'b',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax)

    ax = fig.add_subplot(133)
    #ax.plot(range(1,len(diffs)+1), [np.median(x) for x in this_delays3], c='k')
    bp = ax.boxplot(this_delays3, labels=[-x for x in diffs], widths=0.32)
    #for i, patch in enumerate(bp['boxes']):
    #    patch.set_facecolor(colors2[i%ntypes])
    for i, x in enumerate(this_delays3_median):
        ax.text(1.4+i, x, str(int(round(x))), ha='center', va='center')
    ax.set_xlabel('RASS2 - RASS1 (decrease)')
    ax.yaxis.grid(True)
    ax.set_ylim(ylim)
    ax.set_yticklabels([])
    ax.text(panel_xoffset,panel_yoffset,'c',ha='right',va='top',transform=ax.transAxes,fontsize=11, fontweight='bold')
    seaborn.despine(ax=ax)


    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1)
    if display_type=='pdf':
        plt.savefig('figure2_delays.pdf',dpi=600,bbox_inches='tight',pad_inches=0.01)
    elif display_type=='png':
        plt.savefig('figure2_delays.png',bbox_inches='tight',pad_inches=0.01)
    else:
        plt.show()
