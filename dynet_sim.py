import numpy as np
from scipy.stats import random_correlation,norm 
from scipy.spatial.distance import pdist,squareform
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from pydynet.dynet_statespace import Dynet_SSM

class DynetSim(Dynet_SSM): 
    def __init__(self, n = 5, srate = 200, duration = 2, order = 5, sparsity = .5, nstates = 3, ntrials = 200, snr_db = None, lmix = 0):        
        super().__init__()
        # Variables
        self.n = n
        self.srate = srate
        self.delay = order
        self.sparsity = sparsity
        self.nstates = nstates
        self.ntrials = ntrials
        self.snr_db = snr_db
        self.lmix = lmix
        self.delay = order
        self.time = np.arange(0, duration, 1 / srate)
        self.popt = order + 1
        self.frange = np.arange(1,np.floor(srate/2)+1)
        self.simulated = False
    def simulate(self):
        self.simulated = True
        # Constants
        SClinks = 0.8         # Markov et al. 2012
        ascale = np.arange(0.1,0.51,0.01)
        dt = 1 / self.srate                
        nsamples = len(self.time)
        min_state = np.argmin(np.abs(self.time-0.15))+1  # Minimum duration in frames
        
        
        # - strurctural links (binary mask)
        I = np.eye(self.n)
        UT = np.triu(np.random.choice([0,1],self.n**2,replace=True,p=[1-SClinks,SClinks]).reshape(self.n,self.n))
        MK = UT + UT.T - np.diag(np.diag(UT + UT.T))
        self.SC = MK + I

        # - directed interactions (binary mask)
        FC = np.zeros(self.SC.shape) # HERE USE SC.shape instead of MK.shape
        MKconnections = np.array(np.where(MK))
        num_connections = len(MKconnections[0])
        ids = np.random.choice(num_connections, int(np.fix((1-self.sparsity)*num_connections)),replace=False)
        FC[np.array(MKconnections)[0,ids],np.array(MKconnections)[1,ids]] = 1
        self.FC = FC + I

        # - AR process (univariate)
        self.AR = np.zeros(( self.n, self.n, self.popt, nsamples) ) 

        for i in range(self.n):
            c1 = np.random.choice(ascale,1)
            c2 = (np.max(ascale)-c1)*.95
            self.AR[i,i,0:2,:] = np.tile(np.concatenate((c1,c2)).reshape(1,-1).T,(1,1,nsamples))  # Low Pass
            #self.AR[i,i,0:2,:] = np.tile(np.concatenate((c1,c2)).reshape(1,-1,order='F').T,(1,1,nsamples))  # Low Pass

        # - AR process (univariate)
        cON = np.where(MK*self.FC)
        m = nsamples//min_state
        bf = np.arange(m*min_state).reshape(min_state,m,order='F')
        start_at = np.sort(np.random.choice(np.arange(2,m),self.nstates,replace=False))
        state_ons = bf[0,start_at]
        state_end = np.concatenate([bf[0,start_at[1:]], [nsamples]])
        state_dur = state_end-state_ons


        # Determine states
        self.regimes = dict()
        # Starting (no scaling)
        scalef = 1
        for k in range(self.nstates):
            self.regimes[k] = np.arange(state_ons[k],state_ons[k]+state_dur[k])
            summary = np.ones((len(cON[0]),9))*np.nan
            while True:
                # generate off-diag AR and check stability
                tmpAR = self.AR[:,:,:,self.regimes[k][0]]
                for ij in range(len(cON[0])):
                    i,j = cON[0][ij],cON[1][ij]
                    ij_p = np.random.choice(self.delay,1)[0]
                    ampl1 = np.random.choice(ascale*.5,1)
                    ampl2 = (np.max(ascale*0.5)-ampl1)*0.95
                    osc = np.sign(np.random.randn(2))*np.concatenate((ampl1,ampl2))*scalef
                    summary[ij,:] = np.concatenate(([k,i,j],ampl1,[self.time[self.regimes[k][0]],self.time[self.regimes[k][-1]]], osc, [ij_p]))
                    tmpAR[i,j,ij_p:ij_p+2] = osc
                

                # Stability check
                blockA = np.zeros((self.n*self.popt,self.n*self.popt))
                blockA[:self.n,:] = tmpAR.reshape(tmpAR.shape[0],-1,order='F')
                blockA[self.n:,:(self.popt-1)*self.n] = np.eye((self.popt-1)*self.n) 
                blockA[self.n:,(self.popt-1)*self.n:] = np.zeros(((self.popt-1)*self.n,self.n))

                w,v = np.linalg.eig(blockA)
                if any(abs(w)>.95):
                    scalef = scalef*.95
                else:
                    break
            if k == 0:
                self.summary = np.copy(summary)
            else:
                self.summary = np.concatenate((self.summary,summary),0)

            # Add stable matrices to the dynamical system
            self.AR[:,:,:,self.regimes[k]] = np.tile(tmpAR.reshape(self.n,self.n,self.popt,1),[1,1,1,len(self.regimes[k])])

        # - Data (add nuisance segment at the beginning)
        nuisance = np.min([state_ons[0]*2, int(.5/dt)] )
        self.X = np.zeros((self.ntrials,self.n,nsamples+nuisance))
        ARplus = np.concatenate((self.AR[:,:,:,:nuisance],self.AR),3)
        # simulate between-trials correlation (correlated generative noise)
        CTeigvals = np.random.rand(self.ntrials)
        CTeigvals = CTeigvals/np.sum(CTeigvals)*self.ntrials
        CT = abs(random_correlation.rvs(CTeigvals))*3 
        self.CT = CT.clip(max=1)
        dgI = np.random.choice(np.where(np.eye(self.ntrials).reshape(-1)==0)[0],int((self.ntrials**2-self.ntrials)*.1),replace=False)
        NegCT = np.ones(self.ntrials*self.ntrials)
        NegCT[dgI] = -1
        self.CT *= NegCT.reshape(self.ntrials,self.ntrials) 

        # - Generate time-series
        for k_p in range(self.AR.shape[2]):                   # the actual or p_opt
            self.X[:,:,k_p] = self.CT @ np.random.randn(self.ntrials,self.n)

        self.E = np.copy(self.X)

        for k in range(self.AR.shape[2]+1,nsamples+nuisance):
            innovation = self.CT @ np.random.randn(self.ntrials,self.n) # across trials correlation
            self.E[:,:,k]   = innovation
            self.X[:,:,k]  += innovation
            for l in range(self.AR.shape[2]):
                self.X[:,:,k] += (ARplus[:,:,l,k] @ self.X[:,:,k-l].T).T  

        self.X = self.X[:,:,nuisance:]         # remove nuisance data
        self.Y = np.copy(self.X)               # obseved signal
        self.E = self.E[:,:,nuisance:]         # innovation
        self.AR = self.AR[:,:,:,:nsamples]     # ensure size, AR coeffs

        tmp = self.E.transpose(1,2,0).reshape(self.n,-1,order='F')
        self.R = tmp @ tmp.T
        
        
        # - SNR
        if self.snr_db is not None: 
            self.addnoise('w')
        else: 
            self.noise = 0
            
        # - Linear mixing
        if self.lmix > 0: 
            x = np.random.choice(np.arange(1,151),self.n,replace=False) # 2d lattice 15x15cm 
            y = np.random.choice(np.arange(1,151),self.n,replace=False)
            mixf = norm.pdf(np.arange(0,151),0,self.lmix)
            xy = np.concatenate(([x], [y]),axis=0)
            self.DM = squareform(pdist(xy.T))
            self.LMx = norm.pdf(self.DM,0,self.lmix) / np.max(mixf)
            self.Y = np.matmul(np.tile(self.LMx,[self.ntrials,1,1]),self.Y)
        else: 
            self.DM =  np.ones((self.n,self.n))*np.nan
            self.LMx = np.zeros((self.n,self.n))            
        self.scaling = scalef
        
        
    def addnoise(self, distribution):
        """
        Add white or 1/f noise
            INPUTS: 
                - distribution:  'w' (white noise) or '1/f' pink noise                      
        """
        
        As = np.std(self.Y)
        Aw = As / ( 10**( self.snr_db /20 ) ) 


        if distribution == "w": 
            if len(self.Y.shape) == 3: 
                self.noise = Aw * np.random.randn(self.Y.shape[0],self.Y.shape[1],self.Y.shape[2])
            elif len(self.Y.shape) == 2: 
                self.noise = Aw * np.random.randn(self.Y.shape[0],self.Y.shape[1])

        elif distribution == "1/f":
            print('1/f additive noise has not been implemented yet')
            self.noise = 0            
        self.Y += self.noise
    
    def review(self):
        if not self.simulated:
            raise Exception(" First need to simulate --> use .simulate()")
        else:
            cmap = LinearSegmentedColormap.from_list('connections', [[0,0,0],[.85,.325,.098]])
            fig, axs = plt.subplots(2,2, figsize = (15,15))
            axs[0,0].pcolormesh(self.SC, edgecolors='white', linewidth=2,cmap = cmap)
            axs[0,0].set(xticks = [], yticks = []), axs[0,0].set_title('Structural connections',fontsize=20)
            axs[0,0].invert_yaxis()
            axs[0,0].set_aspect('equal'), axs[0,0].set_frame_on(False)
            axs[0,1].pcolormesh(self.FC,cmap = cmap,edgecolors='white', linewidth=2),
            axs[0,1].set(xticks = [], yticks = []), axs[0,1].set_title('Functional connections',fontsize=20)
            axs[0,1].invert_yaxis()
            axs[0,1].set_aspect('equal'), axs[0,1].set_frame_on(False)

            axs[1,0].plot(self.time, self.Y.mean(0).T)
            axs[1,0].set_ylim([-np.max(abs(self.Y.mean(0).T))*1.2,np.max(abs(self.Y.mean(0).T))*1.2])
            axs[1,0].set_xlim([np.min(self.time),np.max(self.time)+1/self.srate])
            colors = cm.get_cmap('Pastel1')
            for i in range(self.nstates):
                axs[1,0].axvspan(self.time[self.regimes[i][0]-1],self.time[self.regimes[i][-1]], alpha=0.5, facecolor=colors(i),edgecolor = 'k')
            axs[1,0].set_xlabel('time(s) / states',fontsize=18)
            axs[1,0].set_ylabel('activity (a.u.)',fontsize=18)
            axs[1,0].tick_params(axis='both', which='major', labelsize=18)
            axs[1,0].set_title('Surrogate time-series',fontsize=20)


            def multi_pwelch(matr,srate):
                if len(matr.shape)==3:    
                    for tr in range(matr.shape[0]):
                        for ch in range(matr.shape[1]):
                            if (tr==0) and (ch==0):
                                f,pxx_tmp = welch(matr[tr,ch,:],fs=srate, window='hamming',nperseg=np.ceil(len(matr[tr,ch,:])/8),scaling='density', return_onesided=True, detrend=False)
                                pxx = np.zeros((matr.shape[0],matr.shape[1],len(pxx_tmp)))
                                pxx[tr,ch,:] = pxx_tmp;
                            else:
                                _, pxx[tr,ch,:] = welch(matr[tr,ch,:],fs=srate)
                elif len(matr.shape)==2:
                    for ch in range(matr.shape[0]):
                        if ch==0:
                            f,pxx_tmp = welch(matr[ch,:],fs=srate, window='hamming',scaling='density', return_onesided=True, detrend=False,nperseg=np.ceil(len(matr[ch,:])/8))
                            pxx = np.zeros((matr.shape[0],len(pxx_tmp)))
                            pxx[ch,:] = pxx_tmp;
                        else:
                            _, pxx[ch,:] = welch(matr[ch,:],fs=srate, window='hamming',scaling='density', return_onesided=True, detrend=False,nperseg=np.ceil(len(matr[ch,:])/8))
                return pxx, f

            psd, f = multi_pwelch(self.Y.mean(0), self.srate)
            axs[1,1].plot(f,psd.T);
            axs[1,1].plot(f,10*np.log10(psd.T));
            axs[1,1].set_xlabel('F(Hz)',fontsize=18)
            axs[1,1].set_ylabel('psd(dB)',fontsize=18)
            axs[1,1].tick_params(axis='both', which='major', labelsize=18)
            axs[1,1].set_title('Surrogate psd \n(trial averaged)',fontsize=20)
    
    def __repr__(self):        
        if self.simulated:
            return ">> DynetSim:  \n -n:\t\t%s \n -srate:\t%s \n -delay:\t%s \n -popt:\t\t%s \n -time:\t\t%s \n -frange:\t%s \n -sparsity:\t%s \n -nstates:\t%s \n -trials:\t%s \n -SC:\t\t%s \n -FC:\t\t%s \n -AR:\t\t%s \n -Y:\t\t%s \n -E:\t\t%s \n -CT:\t\t%s \n -R:\t\t%s \n -scaling:\t%s \n -regimes:\t%s \n -DM:\t\t%s \n -LMx:\t\t%s \n -summary:\t%s \n -noise:\t%s \n<<" % (self.n, self.srate, self.delay, self.popt, self.time.shape, self.frange.shape, self.sparsity, self.nstates, self.ntrials, self.SC.shape, self.FC.shape, self.AR.shape, self.Y.shape, self.E.shape, self.CT.shape, self.R.shape, self.scaling, len(self.regimes), self.DM.shape, self.LMx.shape, self.summary.shape, self.noise.shape)
        else:
            return ">> DynetSim:  \n -n:\t\t%s \n -srate:\t%s \n -delay:\t%s \n -popt:\t\t%s \n -time:\t\t%s \n -frange:\t%s \n -sparsity:\t%s \n -nstates:\t%s \n -trials:\t%s  \n<<" % (self.n, self.srate, self.delay, self.popt, self.time.shape, self.frange.shape, self.sparsity, self.nstates, self.ntrials)