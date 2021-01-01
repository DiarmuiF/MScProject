
# coding: utf-8

# Script to generate a matrix for correcting spectral stray light
# in hyperspectral imagers

# Diarmuid Finnan
# August 2018

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import time
import copy

def spline_that(x,y,x_new):
#     f = interp1d(x,y)
#     y_new = f(x_new)
    spl = UnivariateSpline(x,y,s=0)
    y_new = spl(x_new)
    
    return y_new



def make_D(data,N,range_start,range_stop):

    # Gap between points for better area measurements
    # Also used as dx
	increment = 1e-3
	# Detector Range
	Det_range = np.arange(range_start,range_stop,increment)

	if len(Det_range) % N != 0:
	    
	    raise Exception("Range size not evenly divisible by N")

	# Shape of datacube
	# ACT
	X = data.shape[2]
	# ALT
	Y = data.shape[1]
	# Wavelength
	Z = data.shape[0]

	Z_range = np.arange(0,Z,1)

	X_slice = round(X/2)
	Y_slice = round(Y/2)
	Z_slice = round(Z/2)

	Y_step = round(Y/N)
	Z_step = round(Z/N)

	BASE = 0
	WIDTH = (max(Det_range)-min(Det_range))/N

	DN_range = np.arange(0,N,1)

	Z_new = np.linspace(0,Z,len(Det_range))
	new_step  = int(len(Z_new)/N)

	if N == 40:
	    Band_def = np.loadtxt("Band_def_40_bands.txt",skiprows=4)
	    
	elif N == 45:
	    Band_def = np.loadtxt("Band_def_45_bands.txt",skiprows=4)

	elif N == 50:
	    Band_def = np.loadtxt("Band_def_50_bands.txt",skiprows=4)

	else:
	    raise Exception("Choose N as either 40,45 or 50")
	    
	peak_pixels = []
	centre_wavelengths = np.ones(N)

	for i in range(N):
		# Datacube used has binned pixels so val/2
	    pixel = int((Band_def[i][0]/2))
	    peak_pixels.append(pixel)
	    # Use real peak location 
	    cw_exact = Band_def[i][2]
	    centre_wavelengths[i] = cw_exact

	# Centre wavelengths of bands based on user defined N and range
	#IB_wavelengths = np.linspace(range_start,range_stop,N)

	# Array for storing SRFs
	# N SRFs each Z long
	SRF_arr = np.zeros((N,Z))

	k = 0
	# Gather SRFs as they are in Hypercube
	# Gather and save SRFs
	plt.figure(figsize=(10,8))
	for i in reversed(peak_pixels):
	    SRF = data[:,i,X_slice]
	    
	    SRF_arr[k] = SRF
	    k+=1


	IB_val_arr = np.zeros(N)

	new_SRF_arr = np.zeros((N,len(Z_new)))

	    
	D = np.zeros((N,N))   

	k = -1 

    # Spline SRFs and get aj
	for i in range(N):
	    temp_SRF = copy.deepcopy(SRF_arr[i])
	    long_SRF = spline_that(Z_range,temp_SRF,Z_new)
	    long_SRF[long_SRF<0] = 0
	    new_SRF_arr[i] = long_SRF

	    IB_val = long_SRF[int((centre_wavelengths[k]-range_start)*(1/increment))]
	    IB_val_arr[i] = IB_val
	    k-=1

	for i in range(N):
	    Di = np.zeros(N)
	    
	    k = 0
	    
	    SRF_i = new_SRF_arr[i]
	    
	    for j in range(0,len(long_SRF),new_step):
	        SRF_area = np.trapz(SRF_i[j:j+new_step],dx=increment)
	        dij = SRF_area/(IB_val_arr[k]*WIDTH)
	        
	        Di[k] = dij
	        k += 1
	    
	    D[i] = Di

	#IB_val_arr = IB_val_arr/max(IB_val_arr)

	return [D,IB_val_arr,WIDTH]

def inversion_correction(D,Y_meas):
	C = np.linalg.inv(D)

	Y_corrected = np.matmul(C,Y_meas)

	return Y_corrected


def tikhonov_correction(D,Y_meas,mylamb):
	DT = np.transpose(D)

	myIdent = np.identity(len(D))

    # Y_corrected = (D^T . D + lamb^2 . I)^(-1) . D^T . Y_meas
	p1 = np.linalg.inv(np.matmul(DT,D) + (mylamb**2)*myIdent)
	p2 = np.matmul(DT,Y_meas)

	Y_corrected = np.matmul(p1,p2)

	return Y_corrected

def convert2radiances(Y_corrected,IB_val_arr,WIDTH):

	Y_corrected_s = Y_corrected/(IB_val_arr*WIDTH)

	return Y_corrected_s


if __name__ == "__main__":
	# Test data with full instrument setup
	data = np.load("20170504-response-left.npy")

	# Number of bands
	N = 40

	range_start = 405
	range_stop = 1001
	D = make_D(data,N,range_start,range_stop)[0]

	timestring = time.strftime('%Y%m%d%H%M%S')

	np.save('D_'+timestring,D)

