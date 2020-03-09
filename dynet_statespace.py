import numpy as np

class Dynet_SSM(object): 
    def __init__(self, AR = None, R = None, PY = None,c=None, FFthr=None):
        # Variables
        self.AR = AR
        self.R = R
        self.PY = PY
        self.FFthr = FFthr
        self.c = c
        
def dynet_SSM_KF(Y, p, uc):
    
    """ Kalman filter for state-space modeling of physiological time series
    --------------------------------------------------------------------------
     INPUTs:
     -Y:       3d array: [Trials X Channels X Time]
               Sample data
               Classical for Trials = 1; General for Trials > 1 [1]
     -p:       positive scalar 
               p-order
     -uc:      update Constants 0<=c<=1
               [c C], C=c if len(uc)==1
    --------------------------------------------------------------------------
     OUTPUT:   KF, structure with fields:
     -AR:      4d array: [Channels X Channels X p X Time]
               MVAR model coefficients
     -R:       3d array: [Channels X Channels X Time]
               Measurement Noise Covariance Matrix (W in [1])
               
     -PY:      3d array: [Trials X Channels X Time]               
               One-step Predictions on Y
     -PYe:     3d array: [Trials X Channels X Time]            
               One-step Residuals
     -c:      self-tuning c for each k
    ==========================================================================
     References:
    
     [1] Milde, T., Leistritz, L., ..., & Witte, H. (2010).
         A new Kalman filter approach for the estimation of high-dimensional
         time-variant multivariate AR models and its application in analysis
         of laser-evoked brain potentials. Neuroimage, 50(3), 960-969.
     [2] Gelb, A. (Ed.). (1974). Applied optimal estimation. MIT press.
    
    --------------------------------------------------------------------------
     Notes: all equations are based on [1], but using the standard variable
     naming in Kalman filters [2]
    """

    # -check input
    if len(np.shape(Y)) < 3:
        raise Exception('Check the dimensions of Y. Y should be in the format [trials, channels, time]')

    if np.size(uc)==1:
        C = uc; 
        c = uc

    # -Preallocate main variable
    trl,dim,tm   = Y.shape
    AR           = np.zeros((dim,dim*p,tm))          # AR matrix
    R            = np.zeros((dim,dim,tm))            # Estimated noise measurement
    PY           = np.zeros(Y.shape)                 # One-step predictions

    # -Initialize unknown design variables
    Rk           = np.eye(dim)*1e-2                  # Noise measurement matrix
    xm           = np.zeros((dim*p,dim)) + 1e-4      # Prior state estimates
    Pmi          = np.eye(dim*p)*1e-4                # Estimated prediction error

    
    # -RECURSION
    #   loop from p(lag)+1 through time
    for k in range(p+1,tm):
        
        # Select previous data for the matrix H        
        data_back    = Y[:,:,k-1:k-p-1:-1]
        H            = np.reshape(data_back,[trl,dim*p],order='F')            # see eq. (6)    
        Z            = Y[:,:,k]    # current observation      

    
        # Recursion on measurement noise covariance W
        Rn           = (Z-H @ xm).T @ (Z-H @ xm) / max([trl-1,1])
        Rk           = Rk + c * (Rn-Rk) # adaptation constant on the past of R   % see eq. (13)
        # One-Step Prediction
        PY[:,:,k]    = H @ xm

        # Innovation or residual covariance
        S            = H @ Pmi @ H.T + np.sum(np.diag(Rk)) * np.eye(trl)       # see eq. (6b)
        # Optimal Kalman Gain
        
        K = np.linalg.solve(S.T, (Pmi @ H.T).T).T    # Pmi @ H.T / S           # see eq. (7)
        # A posteriori state estimate
        xp           = xm + K @ (Z - H @ xm)                                   # see eq. (8)
        xm           = xp           # update xm                                # see eq. (9)

                # A posteriori estimate prediction error covariance
        Ppl         = Pmi - K @ S @ K.T                                        # see eq.(10)
        Pmi         = np.copy(Ppl)          # update Pmi (a priori P)
        Pmi[np.diag_indices(Pmi.shape[0])] += C**2                                                        # see eq.(11)

        #Collect recursive estimates
        AR[:,:,k]   = xm.T
        R[:,:,k]    = Rk

    # -Saving output variables
    AR           = AR.reshape([dim,dim,p,tm],order='F')
    KF           = Dynet_SSM(AR = AR, R = R, PY = PY, c = uc)
    
    return KF

def dynet_SSM_STOK(Y,p,ff):
    """
     The Self-Tuning optimized Kalman filter
    --------------------------------------------------------------------------
     INPUTs:
     -Y:       3d array: [Trials X Channels X Time]
               Sample data
               Classical for Trials = 1; General for Trials > 1 [1]
     -p:       positive scalar 
               p-order
     -ff:     percentage of variance explained for setting the filter factor
               Regularization parameter, default 0.99
    --------------------------------------------------------------------------
     OUTPUT:   STOK, structure with fields:
     -AR:      4d array: [Channels X Channels X p X Time]
               MVAR model coefficients
     -R:       3d array: [Channels X Channels X Time]
               Measurement Noise Covariance Matrix (innovation based)
               
     -PY:      3d array: [Trials X Channels X Time]               
               One-step Predictions on Y
     -c:      self-tuning c for each k
     -FFthr   filtering factor threshold for each k [2]
    ==========================================================================
     References:
     [1] Nilsson, M. (2006). Kalman filtering with unknown noise covariances.
         In Reglerm�te 2006.
     [2] Hansen, P. C. (1987). The truncated svd as a method for 
         regularization. BIT Numerical Mathematics, 27(4), 534-553.
"""
    # -check input
    
    if len(np.shape(Y)) < 3:
        raise Exception('Check the dimensions of Y. Y should be in the format [trials, channels, time]')

    # -Preallocate main variable
    trl,dim,tm   = Y.shape
    AR           = np.zeros((dim,dim*p,tm))          # AR matrix
    R            = np.zeros((dim,dim,tm))            # Estimated noise measurement
    PY           = np.zeros(Y.shape)                 # One-step predictions

    # -Initialize unknown design variables
    xm           = np.zeros((dim*p,dim)) + 1e-4      # Prior state estimates
    trEk         = np.zeros(tm)                      # Innovation monitoring
    allc         = np.zeros(tm)                      # Self-tuning c, for all k
    FFthr        = np.zeros(tm)                      # Spectral decomposition cut-off

    #   loop from p(lag)+1 through time
    for k in range(p+1,tm):

        # Select previous data for the matrix H        
        data_back    = Y[:,:,k-1:k-p-1:-1]
        H            = np.reshape(data_back,[trl,dim*p],order='F')            # see eq. (6)    

        # Measurement and Innovation
        Z            = Y[:,:,k]    # current observation      
        PY[:,:,k]    = H @ xm
        vk           = Z - PY[:,:,k]       # Innovation at time k

        # Measurements innovation monitoring
        tmp          = vk.T @ vk
        R[:,:,k]     = tmp / max([trl-1,1])
        trEk[k]      = np.trace(tmp)

        # SVD Tikhonov (Spectral decomposition)
        # Economy-size decomposition of trl-by-dim H
        u,s,vh       = np.linalg.svd(H,full_matrices=False)
        # only the first "dim" columns of U are computed, and S is [dim,]

        # determine filtering factor threshold [2]
        relv         = s**2 / sum(s**2)
        filtfact     = np.where(np.cumsum(relv) < ff)[0]
        if len(filtfact)>0:
            filtfact = filtfact[-1]
        else:
            filtfact = 0
            
        lambda_k     = s[filtfact]**2
        FFthr[k]     = filtfact
        D = np.diag(s / (s**2 + lambda_k))
        Hinv         = vh.T @ D @ u.T
        betas        = Hinv @ Z

        # self-tuning adaptation constant
        if k > (p+1)**2: # 0.05 <= c <= 0.95
            ntrEk    = trEk[k-1:k-p*2:-1]
            e_k      = np.mean(ntrEk[:p])
            e_p      = np.mean(ntrEk[p+1:])
            x        = abs(e_k-e_p) / e_p
            c        = np.min([0.05 + x, 0.95])
        else:
            c        = 0.05

        allc[k] = c

        # Kalman update
        xp         = (xm + c * betas) / (1+c)    # [1]
        xm         = np.copy(xp)

        #Collect recursive estimates
        AR[:,:,k]    = xm.T

    # -Saving output variables
    AR           = AR.reshape([dim,dim,p,tm],order='F')
    STOK         = Dynet_SSM(AR = AR, R = R, PY = PY, c=allc, FFthr=FFthr)
    return STOK