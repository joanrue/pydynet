# pydynet

The complete collection of python scripts for estimating and simulating
time-varying Multivariate Autoregressive processes (tv-MVAR)
by means of Kalman filtering and Self-Tuning Optimized Kalman filtering.


The toolbox includes:
- One ipython notebook demo (please refer to the file dynet_demo01 for a brief tutorial)
- Four scripts containing classes and functions:
    - 'dynet_statespace'
        - dynet_SSM_KF implements the Kalman filter for state-space modeling of physiological time series
        - dynet_SSM_STOK implements the Self-Tuning Optimized Kalman filter

    - 'dynet_con'
        - dynet_ar2pdc estimates the tv PDC from tv-AR coefficients
        - dynet_connplot displays connectivity matrices (function of time and
          frequency) for each combination of signals
        - dynet_parpsd estimates the AR coefficients in the frequency domain and the parametric power spectral density of the input signals

    - 'dynet_sim'
        - dynet_sim is the simulation class for tv-MVAR generated surrogate time series
        - dynet_sim.review() displays the 1) structural adjacency matrix, 2) the functional adjacency matrix, 3)surrogate time-series in the time domain,
        4) the power spectral density of surrogate time-series
        
    - 'utilities' contains other functions for the demo. 
