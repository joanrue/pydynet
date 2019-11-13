import numpy as np

def auc_roc(X,Y, ncrit):
    if (not np.isnan(Y[0]).any()) and (len(Y.shape) >2):
        dg = Y.shape[0]
        for ij in range(dg):
            X[ij,ij,:,:] = np.nan;
            Y[ij,ij,:,:] = np.nan;
    X = X[~np.isnan(X)]
    Y = Y[~np.isnan(Y)]
    qrange = np.linspace(0.001,1,ncrit)
    sens = np.zeros(ncrit)
    spec = np.zeros(ncrit)
    thre = np.quantile(Y,qrange)

    for k in range(ncrit):
        sens[k] = np.mean(Y[X>0]>=thre[k])
        spec[k] = np.mean(Y[X==0]<thre[k])
    auc = abs(np.trapz(1-spec,sens))    
    return spec,sens,auc

