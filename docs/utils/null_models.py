import pandas as pd
import numpy as np
from tqdm.auto import tqdm

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
    
    for idx in tqdm(range(0,N_wins)): # For data point
        conn_arr = data_df.iloc[idx].copy().values # Copy the connection values in a given colum as an array 
        np.random.shuffle(conn_arr) # Shuffle the connection values in the array
        null_data_df.loc[len(null_data_df.index)] = conn_arr # Add array to null data frame as a row
    
    null_data_df.columns = ['NullC_'+str(i+1).zfill(5) for i in np.arange(data_df.shape[1])]
    null_data_df.index = data_df.index
    return null_data_df
   
# Phase Randomization Function
# ----------------------------
# By Dan Handwerker
def phase_randomize_onets(ts, startbin=1, numbinshift=-1):
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
        #print(f"Ratio of imaginary/real parts of phase shifted ts {ri_ratio}")
        phaserand_ts_noimag = np.real(np.sign(phaserand_ts))*abs(phaserand_ts)
        return(phaserand_ts_noimag)
        
    else:
        print(f"Warning: Ratio of imaginary/real parts of phase shifted ts might be too large {ri_ratio}")
        # Return rts with what might be a non-trivial imaginary part
        return(phaserand_ts)

# Randomize ROI time series for Null Data
# ---------------------------------------
def phase_randomize(data_df):
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
    null_data_df = pd.DataFrame(index=data_df.index, columns=data_df.columns) # Empty data frame for null data
    
    for ROI in tqdm(list(data_df.columns)): # For each ROI
        ROI_arr = data_df[ROI].copy().values # Copy the ROI time series as an array
        ROI_arr = phase_randomize_onets(ROI_arr) # Randomize ROI time series using phase randomization
        null_data_df[ROI] = ROI_arr # Add array to null data frame as a column
    
    return null_data_df
