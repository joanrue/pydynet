
import numpy as np
import matplotlib.pyplot as plt

def dynet_ar2pdc(KF,srate,freqs,metric = 'sPDC',univ = 0,flow = 1,PSD = 0):
    """
     Obtain PDC, sPDC, info-PDC from tvAR coeffients
                                              M.Rubega, D.Pascucci, 17.10.2018
     Last update: 22.10.2019
    --------------------------------------------------------------------------
     INPUTs
     - KF:        Dynet_SSM object (dynet_statespace.Dynet_SSM())
                  containing the 4d array of AR coefficients
                  [n x n x order x time]
                  and the estimated measurement noise covariance matrix R
                  [n x n x time]
     - srate:     Sampling rate
     - freqs:     Frequency 1d array
     - metric:    see OUTPUT
     - univ:      Remove (0, default) or Keep (1) the diagonal elements
     - flow:      normalization per columns (1) or rows (2)
     - PSD:       (1) Add the normalized parametric PSD on diagonals
                  (0) none (only for graphical purpose)
    --------------------------------------------------------------------------
     OUTPUTs
     - PDC:       4d array: [Nodes X Nodes X Freq X Time]
                  one of
                 'PDC'     % non-squared       Eq. (18) in [2]
                 'sPDC'    % squared           Eq. (7) in [3]
                 'PDCnn'   % non-normalized
                 'sPDCnn'  % squared non-normalized
                 'iPDC'    % info-PDC          Eq. (5.25) in [4]
                 'iPDCs'   % info-PDC squared
    --------------------------------------------------------------------------

    References:
     [1] Milde, T., Leistritz, L., ..., & Witte, H. (2010), Neuroimage, 50(3),
         960-969. A new Kalman filter approach for the estimation of
         high-dimensional time-variant multivariate AR models and its
         application in analysis of laser-evoked brain potentials.
     [2] Baccal�, L. & Sameshima, K. (2001) Biol Cybern, 84 (6), 463�474
         Partial directed coherence: a new concept in neural structure
         determination.
     [3] Astolfi, L., ..., & Babiloni, F. (2006), IEEE Transactions on
         Biomedical Engineering, 53(9), 1802-1812
         Assessing cortical functional connectivity by partial directed
         coherence: simulations and application to real data.
     [4] K. Sameshima and L. A. Baccala, (2014), CRC Press
         Methods in brain connectivity inference through multivariate time
         series analysis.
     [5] Toppi, J., ..., & Astolfi, L. (2013), 35th Annual International
         Conference of the IEEE EMBS, 4346-4349
         The Effect of Normalization of Partial Directed Coherence on the
         Statistical Assessment of Connectivity Patterns: A Simulation Study    
    """
    
    if flow not in [1,2]:
        raise Exception('Check the value of "flow" flow must be a value in either 1 or 2')
        
    # Check R
    nodes,_,order,time = KF.AR.shape
    if (KF.R is not None) and (len(KF.R.shape)<3):
        KF.R = np.transpose(np.tile(KF.R,(time,1,1)),[1,2,0])
    elif KF.R is None: 
        KF.R = np.transpose(np.tile(np.eye(nodes),(time,1,1)),[1,2,0])

    # Transform AR (time-domain) into A(f) (frequency-domain)
    Z = np.exp(-2*np.pi*1j*freqs / srate).reshape(-1,1) ** np.arange(1,order+1).reshape(1,-1)
    A = np.transpose(np.tile(np.eye(nodes,dtype=complex),(len(freqs), time,1,1)),[2,3,0,1])

    for k in range(order):
        tmp  = np.transpose(np.tile(-KF.AR[:, :, k,:], (len(freqs),1,1,1)),(1,2,0,3))
        A    += np.multiply(tmp,Z[:,k].reshape(1,1,-1,1))
        
    if metric == 'PDC': # Eq. (18) in [2]    
        if flow==1: 
            PDC = np.divide(abs(A),np.sqrt(np.sum(abs(A)**2,axis = flow-1 )).reshape(1,A.shape[1],A.shape[2],A.shape[3]))
        if flow==2: 
            PDC = np.divide(abs(A),np.sqrt(np.sum(abs(A)**2,axis = flow-1 )).reshape(A.shape[0],1,A.shape[2],A.shape[3]))

    elif metric == 'sPDC': #  Eq.  (7) in [3]
        if flow==1: 
            PDC = np.divide(abs(A)**2,np.sum(abs(A)**2,axis = flow-1 ).reshape(1,A.shape[1],A.shape[2],A.shape[3]))
        if flow==2: 
            PDC = np.divide(abs(A)**2,np.sum(abs(A)**2,axis = flow-1 ).reshape(A.shape[0],1,A.shape[2],A.shape[3]))

    elif metric == 'PDCnn': 
        PDC = abs(A)

    elif metric == 'sPDCnn': 
        PDC = abs(A)**2


    elif metric == 'iPDC': #  Eq.  (5.25) in [4] -> if R is diagonal, "generalized PDC
        #    For construction, R should be constant over time:
        R = np.mean(KF.R[:,:,np.max([round(KF.R.shape[2]/2),order]):KF.R.shape[2]],axis=2)
        SIGMA = np.linalg.pinv(R)
        w_ii = np.tile(np.diag(R),[nodes, len(freqs),time,1]).transpose([3,0,1,2])
        den2 = np.zeros((nodes,len(freqs),time),dtype = complex)
        for j in range(nodes):
            den1 = np.sum(np.tile(A[:,j,:,:],[10,1,1,1]).transpose([1,0,2,3]) * np.tile(SIGMA,[A.shape[2],A.shape[3],1,1]).transpose([2,3,0,1]),axis=0)
            den2[j] = np.sum(den1 * np.conjugate(A[:,j,:,:]),axis=0)
        den = np.tile(den2,[A.shape[0],1,1,1])
        PDC = abs(A) / np.sqrt(w_ii * den)

    elif metric == 'iPDCs': #  Eq.  (5.25) in [4] -> if R is diagonal, "generalized PDC
        #    For construction, R should be constant over time:
        R = np.mean(KF.R[:,:,np.max([round(KF.R.shape[2]/2),order]):KF.R.shape[2]],axis=2)
        SIGMA = np.linalg.pinv(R)
        w_ii = np.tile(np.diag(R),[nodes, len(freqs),time,1]).transpose([3,0,1,2])
        den2 = np.zeros((nodes,len(freqs),time),dtype = complex)
        for j in range(nodes):
            den1 = np.sum(np.tile(A[:,j,:,:],[10,1,1,1]).transpose([1,0,2,3]) * np.tile(SIGMA,[A.shape[2],A.shape[3],1,1]).transpose([2,3,0,1]),axis=0)
            den2[j] = np.sum(den1 * np.conjugate(A[:,j,:,:]),axis=0)
        den = np.tile(den2,[A.shape[0],1,1,1])
        PDC = abs(A)**2 / (w_ii * den)
        
    # -Remove or keep the autoregressive component
    if univ==0:
        for dg in range(PDC.shape[0]):
            PDC[dg,dg,:,:] = np.nan
        
    # -If required, store the parametric PSD on the diagonal 
    # ( only for plotting purposes)
    
    if PSD:
        S_ft = np.median(KF.R[:,:,round(time/2):],axis=2)
        KF.SS = np.zeros((nodes,nodes,len(freqs),time),dtype=complex)
        for t in range(time):
            for f in range(len(freqs)):
                A_ft = A[:,:,f,t]
                KF.SS[:,:,f,t] = np.linalg.solve(A_ft, np.linalg.solve(A_ft,S_ft).T).T
        
        KF.AuS = np.zeros((nodes,len(freqs),time),dtype=complex)
        for k in range(nodes):
            KF.AuS[k,:,:] = KF.SS[k,k,:,:]      

    return PDC
          
def dynet_parpdc(KF,srate,f_range,SSout=2,t_win=None):
    """
     Get A(f) and SS(optional) from AR coefficients
     adds frequency-related info on the Kalman filter estimates     
    --------------------------------------------------------------------------
     INPUTs
     - KF:         Dynet_SSM object (dynet_statespace.Dynet_SSM())
     - srate:      Sampling rate
     - f_range:    Frequency range for A(f) and SS computation, 1d array
                   - linear spacing if len(np.shape(f_range))==2
                   - user-specified spacing if len(np.shape(f_range))>2
     - SSout:      (0) A(f),  (1) A (f ) and SS, (2) SS
     - t_win:      Temporal window (in samples) to restrict A(f) and SS
    --------------------------------------------------------------------------
     OUTPUTs (additional fields in KF)
     - AF:         A(f) matrix
                   4d array: [Channels X Channels X Freq X Time]
     - SS:         Spectral-Autospectral matrix (optional)
                   4d arrray: [Channels X Channels X Freq X Time]
     - AuS:        Autospectral matrix (optional)
                   3d array: [Channels X Freq X Time]
    """

        # Check R
    nodes,_,order,time = KF.AR.shape
    if (KF.R is not None) and (len(KF.R.shape)<3):
        R = np.transpose(np.tile(KF.R,(time,1,1)),[1,2,0])
    elif KF.R is None: 
        R = np.transpose(np.tile(np.eye(nodes),(time,1,1)),[1,2,0])

    if t_win is not None: 
        PAR = KF.AR[:,:,:,t_win]
        R = R[:,:,t_win]
        KF.t_win.f = t_win
        time -= t_win
        
    # Transform AR (time-domain) into A(f) (frequency-domain)
    Z = np.exp(-2*np.pi*1j*freqs / srate).reshape(-1,1) ** np.arange(1,order+1).reshape(1,-1)
    A = np.transpose(np.tile(np.eye(nodes),(len(freqs), time,1,1)),[2,3,0,1])

    for k in range(order):
        tmp  = np.transpose(np.tile(-PAR[:, :, k,:], (len(freqs),1,1,1)),(1,2,0,3))
        A    += np.multiply(tmp,Z[:,k].reshape(1,1,-1,1))
    
    KF.Af = A
    
    if SSout > 0:
        S_ft = np.median(R[:,:,round(time/2):],axis=2)
        KF.SS = np.zeros((nodes,nodes,len(freqs),time),dtype=complex)
        for t in range(time):
            for f in range(len(freqs)):
                A_ft = A[:,:,f,t]
                KF.SS[:,:,f,t] = np.linalg.solve(A_ft, np.linalg.solve(A_ft,S_ft).T).T
        if SSout < 2:
            KF.AuS = np.zeros((nodes,len(freqs),time),dtype=complex)
            for k in range(nodes):
                KF.AuS[k,:,:] = SS[k,k,:,:]

def dynet_connplot(ConnMatrix,time,freq,labels=None,quantrange=[0.01, 0.99],cmap = 'jet',SC = None,univ = 0):
    """
     Figure displaying connectivity matrices (function of time and frequency) 
     for each combination of signals
    --------------------------------------------------------------------------
     INPUTs
     - ConnMatrix:       4d array: [Nodes X Nodes X Freq X Time] 
     - time:             Time vector, 1d array: [1 X Time]
     - freq:             Frequencies vector, 1d array: [1 X Freq]
     - labels:           Nodes' labels np.arange(len(nodes)), by default) 
     - quantrange:       ([.01 .99], by default)
     - cmap:             Colormap colors (jet, by default)
     - SC:               No structural matrix info (0, by default), otherwise
                         structural matrix, 2d array [Nodes X Nodes]
     - univ:             Remove (0, by default) diagonal plots otherwise keep (1)
    --------------------------------------------------------------------------
    """
    # - define labels for each node
    if labels is None:
        labels=['n{}'.format(i+1) for i in range(ConnMatrix.shape[0])]
    
    # - define range of values to display
    dim = ConnMatrix.shape
    if (len(quantrange)>1) and (quantrange[0]>=0):
        maxscale = np.nanquantile(ConnMatrix,quantrange[1])
        minval = np.nanmin(ConnMatrix)
        if minval< 0:
            minscale = np.nanquantile(ConnMatrix, quantrange(1))
        else:
            minscale = 0
    elif len(quantrange>1) and (quantrange[0]<0):
        minscale = quantrange[0]
        maxscale = quantrange[1]
    else:
        minscale = 0
        maxscale = quantrange
    
    dt = (time[1]-time[0])/2.
    df = (freq[1]-freq[0])/2.
    extent = [time[0]-dt, time[-1]+dt, freq[0]-df, freq[-1]+df]
    
    # - plot figure
    
    fig, axs = plt.subplots(nrows = dim[0], ncols = dim[1], sharex=True, sharey= True, figsize = (2*dim[1],2*dim[0]))
    for i1 in range(dim[0]):
        for i2 in range(dim[1]):    
            if univ == 0:
                if i1!=i2:
                    im = axs[i1,i2].imshow(ConnMatrix[i1,i2,:,:], extent = extent, aspect='auto', cmap = cmap, vmin = minscale, vmax = maxscale)
                    axs[i1,i2].axvline(x = 0, linewidth=4, color='w')
                    axs[i1,i2].invert_yaxis()
            else:
                if np.sum(ConnMatrix[i1,i2,:,:]) is not np.nan:
                    im = axs[i1,i2].imshow(ConnMatrix[i1,i2,:,:], extent = extent, aspect='auto', cmap = cmap, vmin = minscale, vmax = maxscale)
                    axs[i1,i2].axvline(x = 0, linewidth=4, color='r')
                    axs[i1,i2].invert_yaxis()

            if (i1 == dim[0]-1) and (i2==dim[1]-1):
                pass#axs[i2,i1].set_axis_off()  

            if (i1 != 0) and (i2 == 0):
                axs[i1,i2].set_ylabel(labels[i1],rotation=45,fontsize=20)              

            if i1 ==0:
                axs[i1,i2].set_xlabel(labels[i2],rotation=45,fontsize=20)
                axs[i1,i2].xaxis.set_label_position('top')

            if (i2 == i1) and (univ==0):
                axs[i1,i2].set_axis_off()      

            if (i1 == 0) and (i2 == dim[1]-1):
                axs[i1,i2].set_ylabel(labels[0],rotation=45, fontsize=20)
                axs[i1,i2].yaxis.set_label_position('right')

            if (i1 == dim[0]-1) and (i2 == 0) and (univ==0):
                axs[i1,i2].xaxis.set_label_position('bottom')

            if (i1 == dim[0]-2) and (i2 == dim[1]-1) and (univ==0):
                axs[i1,i2].set_xlabel('time (s)', fontsize=17)
                axs[i1,i2].set_ylabel('f (Hz)', fontsize=17)
                axs[i1,i2].grid(linestyle='-', linewidth='0.5', color='k')
            else:
                axs[i1,i2].set(xticks=[],yticks=[])

            if univ and (i1+i2==0):
                axs[i1,i2].set_ylabel('f (Hz)', fontsize=17)
                axs[i1,i2].grid(linestyle='-', linewidth='0.5', color='k')
                axs[i1,i2].set(xticks=[])

            if univ and (i1==dim[0]-1) and (i2 ==dim[1]-1):
                axs[i1,i2].set_xlabel('time (s)', fontsize=17)
                axs[i1,i2].grid(linestyle='-', linewidth='0.5', color='k')
                axs[i1,i2].set(yticks=[])

            [axs[i1,i2].spines[i].set_linewidth(4) for i in axs[i1,i2].spines]
            if SC is not None:
                if (SC[i1,i2] == 1) and (i1!=i2):
                    [axs[i1,i2].spines[i].set_color('r') for i in axs[i1,i2].spines]

    fig.tight_layout()
    bbox_ax = axs[dim[0]-1,dim[1]-1].get_position()

    cbar_ax = fig.add_axes([1.01, bbox_ax.y0, 0.02, bbox_ax.y1*(np.max([1,dim[1]/2]))-bbox_ax.y0])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.set_frame_on(True)
    [cbar.ax.spines[i].set_linewidth(4) for i in cbar.ax.spines]
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.tick_params(labelsize=20)
    plt.show()