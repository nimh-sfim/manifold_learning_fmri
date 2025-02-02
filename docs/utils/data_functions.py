#! /usr/bin/env python
# Isabel Fernandez 02/22/2022

# This file contains functional that computes the sliding window correlation matrix.

import pandas as pd
import numpy as np
import os.path as osp
from tqdm.auto import tqdm

# Compute Sliding Window Correlation
# ----------------------------------
def compute_SWC(ts,wl_trs,ws_trs,win_names=None,window=None, keep_progress_bar=True):
    """
    This function will perform the following actions:
    1) Generate windows based on length, step and TR. This means computing window onsets and offsets
    2) Generate window names if those are not provided
    3) For each sliding window:
       * extract time series for all ROIs
       * multiply by the provided window shape
       * compute connectivity matrix
       * extract top triangle
       * apply fisher-transform
       
    INPUTS
    ------
    ts: (array) ROI timeseries in the form of a pd.DataFrame
    wl_trs: (int) window length in number of TRs
    ws_trs: (int) window step in number of TRs
    win_names: window labels as string array. If empty, labels will be generated automatically
    window: (np.array of length equal to wl_trs) window shape to apply
    
    OUTPUTS
    -------
    swc_r: (pd.Dataframe) sliding window connectivity matrix as Pearson's correlation
    swc_Z: (pd.Dataframe) sliding window connectivity matrix as Fisher's transform
    winInfo: (dict) containing window onsets, offsets, and labels.
    """

    [Nacq,Nrois] = ts.shape
    winInfo             = {'durInTR':int(wl_trs),'stepInTR':int(ws_trs)} # Create Window Information
    winInfo['numWins']  = int(np.ceil((Nacq-(winInfo['durInTR']-1))/winInfo['stepInTR'])) # Computer Number of Windows
    winInfo['onsetTRs'] = np.linspace(0,winInfo['numWins'],winInfo['numWins']+1, dtype='int')[0:winInfo['numWins']] # Compute Window Onsets
    winInfo['offsetTRs']= winInfo['onsetTRs'] + winInfo['durInTR']
    
    # Create Window Names
    if win_names is None:
        winInfo['winNames'] = ['W'+str(i).zfill(4) for i in range(winInfo['numWins'])]
    else:
        winInfo['winNames'] = win_names
    
    # Create boxcar window (if none provided)
    if window is None:
        window=np.ones((wl_trs,))
    
    # Compute SWC Matrix
    for w in tqdm(range(winInfo['numWins']),desc='Window',leave=keep_progress_bar):
        aux_ts          = ts[winInfo['onsetTRs'][w]:winInfo['offsetTRs'][w]]
        aux_ts_windowed = aux_ts.mul(window,axis=0)
        aux_fc          = aux_ts_windowed.corr()
        sel             = np.triu(np.ones(aux_fc.shape),1).astype(np.bool)
        aux_fc_v        = aux_fc.where(sel)

        if w == 0:
            swc_r  = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
        else:
            new_df = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
            swc_r  = pd.concat([swc_r,new_df],axis=1)
    swc_Z = swc_r.apply(np.arctanh)
    
    return swc_r, swc_Z, winInfo


# Phase Randomization Function
# ----------------------------
# By Dan Handwerker
def phase_randomize(ts, startbin=1, numbinshift=-1):
    """
    Take a time series and phase randomize it

    INPUTS
    ------
    ts: A 1D time series
    With default parameters, all frequencies will be phase randomized
    
    This function can also randomize a subset of frequencies
    startbin: The first frequency bin to randomize. Note 0 is the DC bin
      which cannot have a phase shift so the minimum value is 1
    numbinshift: Number of bins to shift. If this is 5 that means frequency bins
      1 to 6 are shifted. Default = -1, which means randomize everything

    OUTPUTS
    -------
    phaserand_ts: The phase shifted time series.
       NOTE: If there should be a floating-point error's worth of imaginary values
       in the output. If that's true, then only the real part is returned. If the output
       is complex (something probably went wrong) this will return complex values.
    """


    #print(f"time series length is {len(ts)}")

    # take the fft and separate it into magnitude & phase
    ffts = np.fft.fft(ts)
    mag_ffts = abs(ffts)

    # I originally wrote this code to be able to phase randomize
    #  only a subset of frequency bins. For that, I need to get
    #  the phase of each value or retain the phase where I'm not
    #  randomizing. I'm using cmath.phase for this, which operates 
    #  on one value at a time. There's probably a numpy function 
    #  that gets the phase for all complex values (and this math 
    #  isn't hard to do manually) I was only using this on one 
    #  times series, but if you're doing this a lot, you'll want 
    #  to improve this
    # I've added an if clause that says, if you're randomizing
    #  everything, then ph_ffts can just be a zero array since it
    #  will be filled in with random values.
    ph_ffts = np.zeros(len(ffts))
    if numbinshift>0 or startbin>1:
        # cmath.phase 
        for idx in range(len(ffts)):
            ph_ffts[idx] = cmath.phase(ffts[idx])

    if numbinshift<0:
        numbinshift = np.floor((len(ts)-startbin)/2)

    numbinshift = round(numbinshift) # Should already be an integer, but must be int datatype

    if (startbin+numbinshift-1)>(len(ts)/2):
        raise ValueError(f"startbin {startbin} + numbinshift {numbinshift} must be less than half the number of values in the time series {len(ts)}")


    # For phase randomization, if the inverse FFT will results in real rather than
    #   complex values, then the phase values need to have conjugate symmetry
    #   I'm calculating random phase values for half of the FFT bins and
    #   flipping them for the other half
    randphase = 2*np.pi*np.random.random_sample(size=numbinshift)-np.pi
    ph_ffts[startbin:(startbin+numbinshift)] =  randphase
    ph_ffts[(-startbin):(-startbin-numbinshift):(-1)] = -randphase
    # inverse FFT back to the phase randomized time series
    cffts = mag_ffts * ( np.cos(ph_ffts) + np.sin(ph_ffts)*1j )
    phaserand_ts = np.fft.ifft(cffts)

    # Calculate the real/imaginary ratio for the time series
    # If it's really small, assume this is a floating point error and just return the real part
    # If it's larger, print a warning and return the complex values
    ri_ratio = abs(np.imag(phaserand_ts)).mean()/abs(np.real(phaserand_ts)).mean()
    if ri_ratio < 10e-10:
        print(f"Ratio of imaginary/real parts of phase shifted ts {ri_ratio}")
        phaserand_ts_noimag = np.real(np.sign(phaserand_ts))*abs(phaserand_ts)
        return(phaserand_ts_noimag)
        
    else:
        print(f"Warning: Ratio of imaginary/real parts of phase shifted ts might be too large {ri_ratio}")
        # Return rts with what might be a non-trivial imaginary part
        return(phaserand_ts)

    
# Randomize ROI time series for Null Data
# ---------------------------------------
def randomize_ROI(data_df):
    """
    This function randomwizes the each ROI times series. This function uses the numpy function 
    phase_randomize() function defined above to randomize to the times series data.
    
    INPUTS
    ------
    data_df: (pd.DataFrame) ROI time seres data to randomize (TRs x ROIs)
    
    OUTPUTS
    -------
    null_data_df: (pd.DataFrame) Randomized ROI time series data (TRs x ROIs)
    """
    
    N_trs, N_ROIs = data_df.shape # Save number of times points and ROIs
    null_data_df = pd.DataFrame() # Empty data frame for null data
    
    for ROI in list(data_df.columns): # For each ROI
        ROI_arr = data_df[ROI].copy().values # Copy the ROI time series as an array
        ROI_arr = phase_randomize(ROI_arr) # Randomize ROI time series using phase randomization
        null_data_df[ROI] = ROI_arr # Add array to null data frame as a column
    
    return null_data_df


# Randomize Connections for Null Data
# -----------------------------------
def randomize_conn(data_df):
    """
    This function randomwizes the connections in each window for an SWC matrix.
    This function uses the numpy function np.random.shuffle() to randomize the connection order.
    Note that since the np.random.shuffle() is random the function will output a different null 
    data even with the same input data.
       
    INPUTS
    ------
    data_df: (pd.DataFrame) SWC matrix to randomize (windows x connections)
    
    OUTPUTS
    -------
    null_data_df: (pd.DataFrame) Randomized SWC matrix (windows x connections)
    """
    
    N_wins, N_cons = data_df.shape # Save number of windows and connections
    null_data_df = pd.DataFrame(columns=data_df.columns) # Empty data frame for null data
    
    for idx in range(0,N_wins): # For data point
        conn_arr = data_df.iloc[idx].copy().values # Copy the connection values in a given colum as an array 
        np.random.shuffle(conn_arr) # Shuffle the connection values in the array
        null_data_df.loc[len(null_data_df.index)] = conn_arr # Add array to null data frame as a row
    
    return null_data_df