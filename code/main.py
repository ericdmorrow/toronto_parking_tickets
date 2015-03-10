#!/usr/bin/python

'''
Explore parking ticket data from Open Data Toronto
Eric Morrow
'''

# Module Imports
import os as os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import pypeaks

# General comments about data format
'''
First line is header
Date format is yyymmdd
Some missing data
'''

# Load parking ticket data into a pandas dataframe
print 'Loading data ...'

# Find the data files in the input set
flist = os.listdir('../data/')
flist = [x for x in flist if ('Parking' in x)] 

ind = 1
for fn in flist:
	if ind==1:
		data = pd.read_csv('../data/'+fn)
	else:
		tmp = pd.read_csv('../data/'+fn)
		data = pd.concat([data, tmp])
	ind += 1
				
print data

# Convert date to datetime structure
data['date_of_infraction'] = pd.to_datetime(data['date_of_infraction'], format='%Y%m%d')


##############  Exploratory Data Analysis
# Plot total number of infractions as a function of day

# Resample the data frame for a 1 daily period, keeping track of the total number of infractions
ts = pd.DataFrame(1, index = data['date_of_infraction'], columns=['count'])
rs = ts.resample('1d', how='sum')

plt.figure()
rs.plot()
plt.title('Total number of infractions as a function of time')
#plt.show()

'''
Notable things:
1) There is an apparent periodic structure with a low number of infractions being given on weekends
2) There are also significant downward spikes around hoildays and long weekends
3) There is a noticable decline in the mean number of infractions from Jul-Sep, likely corresponding to vacations 
'''

# Plot percentage of infractions as a function of the day of the week (dow)
rs['dow'] = pd.DataFrame(rs.index.weekday, index=rs.index)

#Compute the total number of infractions on each day
tmp = np.array(rs.groupby('dow').sum()) 

#Normalize by total of all infractions to get a percentage
tmpn = tmp/float(sum(tmp)) 

plt.figure()
plt.bar(range(0,len(tmp)),tmpn)
plt.title('Percentage of total infractions for each day of the week')
#plt.show()

'''
Notable things:
1) Sunday (index 7) accounts for the lowest number of tickets given at 10.5% while Tuesday - Friday account for ~15%
'''

# Fourier transform of number of infractions
# For a little more depth on the apparent periodicity in the structure, we can compute the discrete fourier transform of the
# signal and examine the dominant frequencies

count = np.array(rs['count'])

# A quick plot of the power spectrum shows pinkish so differenitate.  This also deals with the DC shift in the zero frequency component  
count = np.diff(count)

# Compute the discrete fourier transform
dft = sp.fftpack.fft(count)
freq = sp.fftpack.fftfreq(len(dft), 1.)

# Keep only the positive frequecies
ind = freq>0 

# Plot data DFT
plt.figure()
plt.plot(freq[ind], abs(dft[ind]))
plt.grid()
plt.title('DFT of total infraction timeseries')
#plt.show()

# Find the dominant peaks in the spectrum
peaks_data = pypeaks.Data(freq[ind], abs(dft[ind]), smoothness=0.01)
peaks_data.get_peaks(method='slope',avg_interval=1)

print peaks_data.peaks['peaks'][0]

# The get_peaks() method returns the frequency, now find the index in the frequency vector 
idft = np.zeros(len(dft)).astype('complex64')
for peak in peaks_data.peaks['peaks'][0]:
	#positive frequency
	peak_index = pypeaks.slope.find_nearest_index(freq, peak)
	idft[peak_index] = dft[peak_index]
	#negative frequency
	peak_index = pypeaks.slope.find_nearest_index(freq, -peak)
	idft[peak_index] = dft[peak_index]


#Check reconstruction with dominant frequencies (imag components are small so just use the real components)
yfit = sp.fftpack.ifft(idft).real

plt.figure()
plt.plot(count)
plt.plot(yfit,'r')
plt.title('Periodic component fit')
#plt.show()

# Look at the DFT of the residuals
residuals = count - yfit
dftres = sp.fftpack.fft(residuals)

plt.figure()
plt.plot(freq[ind], abs(dftres[ind]))
plt.grid()
plt.title('DFT of periodic fit residuals')
#plt.show()


'''
Notable things:
1) There are 4 dominant components at:
~0.14 day^-1 (corresponding to ~7 day periods) 
~0.35 day^-1 (~2.8 days)
~0.42 day^-1 (~2.3 days)
~0.28 day^-1 (~3.5 days)

'''

# Some of the structure seen within the residuals is not being capture by the periodic mean function.  Try to
# fit residuals with AR model

# Transform the residual distribution to zero mean, unitary standard deviation to make it easier for the fit procedure
res_mu = np.mean(residuals)
res_sig = np.std(residuals)

residuals = residuals - res_mu
residuals = residuals/res_sig

rs['residuals'] = pd.DataFrame(residuals, index=rs.index[1:])

# Find the optimal number of model parameters in AR based on AIC
bestResult = [9999999999, 0]
for i in range(10): #AR model order
	mod = sm.tsa.ARMA(rs['residuals'].values[1:], (i,0)).fit(maxfun=500)
	print i, mod.aic
	
	if mod.aic < bestResult[0]:
		bestResult[0] = mod.aic
		bestResult[1] = i


# Compute the optimal model
mod = sm.tsa.ARMA(rs['residuals'].values[1:], (bestResult[1],0)).fit(maxfun=500)

# Make the predictions using the optimal model
modpred = mod.predict()

# Compare the residuals with the predictions
plt.figure()
plt.plot(residuals)
plt.plot(modpred,'r--')
plt.title('Comparison of residuals and ARMA model')

# Transform model predictions back to original form
modpred = modpred*res_sig + res_mu

# Add back to periodic mean
pred = yfit + modpred

# Compare with differenced total count
fig = plt.figure()
plt.plot(count)
plt.plot(pred, 'r')
plt.title('Comparison beteen the (diff) total number of infractions and model')
plt.show()



