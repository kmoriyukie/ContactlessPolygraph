import filters
import numpy as np
import scipy
from math import pi
import matplotlib.pyplot as plt

fs = 10000
t = np.linspace(0,15,int(1.5*fs))
dim = 1
t = t/10.0
x1 = scipy.signal.sawtooth(2*np.pi*50*t)

L = len(x1)

f = np.linspace(-int(10*L/2) , int(10*L/2)-1,L)/(L*10) * fs

bandpassedSignal = filters.idealBandPassing(x1, 0.5, 1, 1.5)

plt.plot(f,np.abs(scipy.fft.fft(bandpassedSignal)))
plt.show()