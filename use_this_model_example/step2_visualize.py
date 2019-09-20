import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 9})
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')


if __name__=='__main__':
    ## load the results
    
    # ypz:  the predicted z-score (continuous value before discretizing into RASS levels), shape=(#recording, #segment)
    # yp1d: the predicted RASS, shape=(#recording, #segment)
    # yp2d: the predicted probability of RASS, shape=(#recording, #segment, 6), the first column is RASS -5, ...the last columns is RASS 0.
    res = sio.loadmat('RASS_output.mat')
    ypz_RASS = res['ypz']
    yp1d_RASS = res['yp1d']
    yp2d_RASS = res['yp2d']
    ordinal_thres = res['ordinal_thres'].flatten()
    specs_db = res['specs_db']
    freqs = res['freqs'].flatten()
    
    res = sio.loadmat('CAMICU_output.mat')
    yp1d_CAMICU = res['yp1d']
            
    ## visualize the result
    
    display_type = 'png' #show/png/pdf
    K = 6  # currently we only track 6 levels in RASS (-5, -4, ..., 0) [DO NOT CHANGE]
    window_time = 4  # [s] [DO NOT CHANGE]
    window_step = 4  # [s] [DO NOT CHANGE]
    
    # take the first recording
    yp2d_RASS = yp2d_RASS[0]
    yp1d_RASS = yp1d_RASS[0]
    ypz_RASS = ypz_RASS[0]
    yp1d_CAMICU = yp1d_CAMICU[0]
    
    Ts = np.arange(len(yp2d_RASS))*window_time/60.
    plt.close()
    fig = plt.figure(figsize=(12,7))
    
    # RASS
    ax1 = fig.add_subplot(411)
    ax1.step(Ts, yp1d_RASS, color='k')
    ax1.set_yticks([-5, -4, -3, -2, -1, 0])
    ax1.yaxis.grid(True)
    ax1.set_ylabel('RASS')
    ax1.set_ylim([-5.4, 0.4])
    plt.setp(ax1.get_xticklabels(), visible=False)
    seaborn.despine()
    
    # z-score (continuous value before discretizing into RASS levels)
    ax2 = fig.add_subplot(412, sharex=ax1)
    ax2.plot(Ts, ypz_RASS, color='b')
    for i in ordinal_thres:
        ax2.axhline(i, color='k', ls='--')
    ax2.set_ylabel('RASS z-score')
    #ax2.set_ylim([-5.2, 0.2])
    plt.setp(ax2.get_xticklabels(), visible=False)
    seaborn.despine()
    
    """
    # Probability of RASS
    ax3 = fig.add_subplot(513, sharex=ax1)
    cmap = matplotlib.cm.get_cmap('jet')
    colors = [cmap(1-1./5*i) for i in range(len(K))]
    yp2d_RASS_cum = np.c_[np.zeros(len(yp2d_RASS)), np.cumsum(yp2d_RASS, axis=1)]
    for i in range(yp2.shape[1]-1):
        ax3.fill_between(times, yp2[:,i], yp2[:,i+1], facecolor=colors[i], edgecolor='none', alpha=0.7)
    for i in range(K):
        ax3.plot([0,1], [100,100], color=colors[i], lw=4, label='%d'%(i-5,))
    ax3.legend(ncol=K, facecolor='w', frameon=True, framealpha=0.8)
    ax3.set_ylabel('Probability of RASS')
    ax3.set_ylim([0,1])
    ax3.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax3.set_yticklabels(['0','0.2','0.4','0.6','0.8','1'])
    plt.setp(ax3.get_xticklabels(), visible=False)
    seaborn.despine()
    """
    
    # CAMICU
    ax3 = fig.add_subplot(413, sharex=ax1)
    ax3.step(Ts, yp1d_CAMICU, color='k')
    #ax3.set_yticks([-5, -4, -3, -2, -1, 0])
    ax3.yaxis.grid(True)
    ax3.set_ylabel('CAM-ICU')
    ax3.set_ylim([-0.1, 1.1])
    plt.setp(ax3.get_xticklabels(), visible=False)
    seaborn.despine()
    
    #EEG spectrogram
    ax4 = fig.add_subplot(414, sharex=ax1)
    vmin, vmax = np.percentile(specs_db.flatten(), (5,95))
    ax4.imshow(specs_db.mean(axis=2).T, aspect='auto', origin='lower', cmap='jet',
        extent=(Ts.min(), Ts.max(), freqs.min(), freqs.max()),
        vmin=vmin, vmax=vmax)
    ax4.set_xlim([Ts.min(), Ts.max()])
    ax4.set_ylim([freqs.min(), freqs.max()])
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1)
    
    if display_type=='pdf':
        plt.savefig('step2_visualization.pdf',dpi=600,bbox_inches='tight',pad_inches=0.01)
    elif display_type=='png':
        plt.savefig('step2_visualization.png',bbox_inches='tight',pad_inches=0.01)
    else:
        plt.show()
