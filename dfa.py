# Detrended Fluctuation Analysis - measure of self-similarty for time-series analysis. 
# By: Jeev Kiriella (jeevooo)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dfa(signal, box_min, box_max): #Parameters for the defined function
    

        
    ## Plot the input signal ##
    ## Plot the distribution of data points ##
    
    plt.subplot(2, 2, 1)
    plt.plot(signal, color = 'skyblue')
    plt.xlabel('Samples (n)')
    plt.ylabel('Signal Magnitude')
    plt.subplot(2, 2, 2)
    plt.hist(signal, color = 'skyblue', ec = 'black')
    plt.xlabel('Bins')
    plt.ylabel('Count (#)')
    plt.show()
    
    ## Integrate and plot input signal ##
    
    integ_sig = np.array([0]) #Create a numpy array to append results
    i = 1 # Set a counter
    cur_int = 0 
    p = signal.shape[0] # get length of singal input
    
    while i < p + 1:
        cur_int = signal[i] - np.mean(signal) #zero-base the signal
        integ_sig = np.append(integ_sig, integ_sig[i-1] + cur_int) # rectangle method 
        i = i + 1
        if i == p: # Avoid infinite while loop. 
            break
        
    plt.subplot(2, 2, 3)
    plt.plot(integ_sig, color = 'skyblue')
    plt.xlabel('Samples (n)')
    plt.ylabel('Integrated Signal Amplitude (original units)')
    plt.show()
    
    ni = integ_sig.shape[0]
    
    ## Calculate RMS for each box size ##
    
    log_b = np.array([0])
    log_r = np.array([0]) 
    

    for box_size in range (box_min, box_max):
        j = np.floor_divide(ni,box_size) # set number of points per box.
        r = np.array([0])
        
        for i in range(0,j+1):
            z = np.arange(((box_size * i) - box_size), (box_size * i))
            seg = integ_sig[z]
            
            ## Calculate LOBF and RMS for each segment.
            rms = 0
            slope_num = 0
            slope_den = 0
            seg_pred = np.array([0])
            
            for k in range(0,box_size):
                slope_num = slope_num + ((z[k] - np.mean(z)) * (seg[k] - np.mean(z)))
                slope_den = slope_den + ((z[k] - np.mean(z))**2)
            np.seterr(divide='ignore', invalid='ignore')
            slope = slope_num / slope_den
            intercept = np.mean(seg) - (slope * np.mean(z))
            
            for l in range (0,box_size):
                seg_pred = np.append(seg_pred, intercept + (slope * z[l]))
            seg_pred = seg_pred[1:,]
            
            # Detrend the segmet of interest
            for m in range (0,box_size):
                rms = rms + ((seg[m] - seg_pred[m])**2)
            
            # calculate the root-mean square
            rms = rms / box_size
            rms = rms**0.5
            r = np.append(r, rms)
        
        #r = r[1:,]
        r = np.mean(r)
        log_b = np.append(log_b, np.log(box_size))
        log_r = np.append(log_r, np.log(r))
        
    log_b = log_b[9:,]
    log_r = log_r[9:,]
    
    p = np.polyfit(log_b, log_r, 1)
    plt.subplot(2, 2, 4)
    plt.scatter(log_b, log_r, color = 'skyblue')
    plt.plot(np.unique(log_b), np.poly1d(np.polyfit(log_b, log_r, 1))(np.unique(log_b)), color = 'black')
    plt.xlabel('Log Box Size')
    plt.ylabel('Log RMS')
    plt.show()
    
    fractindex = np.round_(p[0], 2)
    
    return print('fractal_index: ', fractindex)


    
