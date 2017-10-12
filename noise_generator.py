# Noise generator - 1/f noise simulation
# Jeev Kiriella (jeevooo)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dfa import dfa

VALID_FRACT = np.arange(0, 2, 0.05) # define the fractindex as a constant global so not repeated on each function use. 

def noise_generator(n, mean, std, fractindex):
    
    """
    1/f Noise simulation based on Kasdin (1995). 
    
    Input (arguments):
     n - number of desired data points (n should be a power of 2) for series (int).
     mean - series mean (int).
     std - series standard deviation (int).
     fractindex - desired fractal index for simulated series (int: range: 0-2).
    
    Output (return):
     x - numpy array of length n with mean, std, fractindex specified. 
     
    """
    if fractindex not in VALID_FRACT:
        raise ValueError("results: status must be one of %r." % VALID_FRACT)
    
    stdev = std
    
    b = 2*fractindex-1
    print('beta: ', b)
    
    bdis = np.zeros(n)

    bdis[0] = 1
    for i in range(1,n):
        bdis[i] = bdis[i-1] * (0.5 * b + (i-1))/i # note that b is the shape parementer (b)

    plt.plot(bdis)
    plt.show

    wnt = np.random.normal(mean, stdev, size = n)
    print('WhiteNoise Stdev: ', np.std(wnt))
    plt.plot(wnt)
    plt.show()

    bdis_freq = np.fft.fft(bdis)
    wnt_freq = np.fft.fft(wnt)

    bdis_freq = bdis_freq[1:n+1]
    wnt_freq = wnt_freq[1:n+1]

    freq_total = bdis_freq * wnt_freq
    
    NumUniquePts = n/2 + 1
    NumUniquePts = int(NumUniquePts)
    j = np.arange(1, NumUniquePts)
    
    if fractindex > 1.0:
        j = j
    elif fractindex <= 1.0:
        j = j**0.5
    
    ft_half1 = freq_total[1:NumUniquePts]/j

    real = np.real(freq_total[1:NumUniquePts+1])
    real = np.flip(real, axis=0)

    imaginary = np.imag(freq_total[1:NumUniquePts+1])
    imaginary = np.flip(imaginary, axis=0)
    imaginary = 1j * imaginary

    ft_half2 = real - imaginary

    ft = np.hstack((ft_half1, ft_half2))
    
    x = np.fft.ifft(ft)
    x = np.real(x[:n])

    mean_diff = mean - np.mean(x)
    x = mean_diff + x
    print(np.mean(x))
    print(np.std(x))
    plt.plot(x)
    plt.show()
    
    return x
    
    
    
    
