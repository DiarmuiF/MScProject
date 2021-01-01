
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy
from scipy.interpolate import UnivariateSpline

# Number of bands
N = 40

# Step size in range
increment = 1e-3

X = np.arange(400,1000,increment)

BASE = 0
WIDTH = (max(X)-min(X))/N

DN_range = np.arange(0,N,1)

# Step for sampling in each band
step = int(len(X)/N)

# # dx value for area measurements
# dx_scale = 1e-2
# mydx = increment*dx_scale

IB_amp = 929

# Functions for testing misalignment of bands
misalign = np.random.randn(N)*2
#misalign = np.zeros(N)
#misalign = np.ones(N)*7
#amplitudes = np.random.randn(N)+5
amplitudes = np.ones(N)

def tophat(x, base_level, hat_level, hat_mid, hat_width):
    '''
    Creates data set of tophat function
    
    :param x: data set of spectrum this will be used with
    :param base_level: value of data either side of tophat
    :param hat_level: amplitude of the hat
    :param hat_mid: value of centre of the tophat
    :param hat_width: the width of the tophat
    ''' 
    return np.where((hat_mid-hat_width/2. < x) & (x < hat_mid+hat_width/2.),
                    hat_level, base_level)

def genSRF(x, centre_loc, BASE, WIDTH):
    
    base = BASE
    width = WIDTH
    

    
    IB = tophat(x,base,IB_amp,centre_loc,width)
    
    oob_amp = 47
    offset = 6*WIDTH
    
    if (centre_loc-(offset+width)) < min(X):
        oob1 = tophat(x,base,oob_amp,centre_loc+offset,width)
        oob2 = tophat(x,base,oob_amp,centre_loc+2*offset,width)

    elif (centre_loc+(offset+width)) > max(X):
        oob1 = tophat(x,base,oob_amp,centre_loc-offset,width)
        oob2 = tophat(x,base,oob_amp,centre_loc-2*offset,width)
    
    else:
        oob1 = tophat(x,base,oob_amp,centre_loc+offset,width)
        oob2 = tophat(x,base,oob_amp,centre_loc-offset,width)


    SRF = IB + oob1 + oob2 
    
    return SRF

def spline_that(x,y,x_new):
#     f = interp1d(x,y)
#     y_new = f(x_new)
    spl = UnivariateSpline(x,y,s=0)
    y_new = spl(x_new)
    
    return y_new

def gen_meas(L,SRF):
    product = L*SRF

    area = np.trapz(product,dx=increment)
    
    return area

def step_func(x):
    '''
    Step function 
    Returns 1 for first half of x and 0 for second half
    '''
    return 587 * (x < x[int(len(x)/2)])

def Gauss(x,a,b,c):
    '''
    Gaussian function for fitting peaks
    
    :param x: data
    :param a: height of peak
    :param b: x location of peak (mean)
    :param c: width of peak (sigma)
    '''
    return a*np.exp(-(x - b)**2 /(2*c**2))

def shorten_set(data,step):
    shortened = []
    for j in range(0,len(data),step):
        temp_range = data[j:j+step]
        mean_val = np.mean(temp_range)
        shortened.append(mean_val)
        
    return np.array(shortened)


# In[2]:


L = step_func(X) + 1

plt.plot(X,L)
plt.show()

# file = 'radiances.txt'
        
# f = open(file)

# rad_data = f.read().split("\n")
# rad_data = rad_data[:-1]

# wavelength = []
# r1 = []
# r2 = []
# r3 = []
# r4 = []

# for lines in rad_data:
#     numbers = lines.split(" ")
#     wavelength.append(float(numbers[0]))
#     r1.append(float(numbers[1]))
#     r2.append(float(numbers[2]))
#     r3.append(float(numbers[3]))
#     r4.append(float(numbers[4]))
    
# r1 = r3[:-1]
# r1 = spline_that(np.arange(0,600,1),r1,np.arange(0,600,increment))
# L = r1

# len_radiances = len(r1)
# radiance_step = round(len_radiances/N)

# plt.figure(figsize=(10,8))
# plt.plot(r1,label="L")
# #plt.plot(ALT,FP_trans_show,label="Ideal Filter Transmission")
# # plt.plot(wavelength,r2,label="water")
# # plt.plot(wavelength,r3,label="vegetation")
# # plt.plot(wavelength,r4,label="soil")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Radiance")
# plt.legend()
# plt.show()

# data = np.load("testdata.npy")

# L = spline_that(np.linspace(400,1000,len(data)),data,X)*50

# plt.plot(L)
# plt.show()


# In[3]:


vary_amp = Gauss(X,1,700,300)

plt.plot(X,vary_amp)
plt.show()

vary_amp = shorten_set(vary_amp,int(len(X)/N))


# In[4]:


IB_wavelengths = np.linspace(min(X)+WIDTH/2,max(X)-WIDTH/2,N)

band_starts = np.arange(min(X),max(X),WIDTH)

if len(X) % N != 0:
    
    raise Exception("X not evenly divisible by N")
    
# Arrays for storing SRFs
SRF_arr = np.zeros((N,len(X)))
SRF_arr_norm = np.zeros((N,len(X)))
ti_arr = np.zeros((N,len(X)))

D = np.zeros((N,N))

IB_area = np.zeros(N)

IB_val_arr = np.zeros(N)

plt.figure(figsize=(10,8))

for i in range(N):
    # Create SRF with satellites and IB wavelength
    # at the centre of a defined bands
    #SRF = genSRF(X,IB_wavelengths[i]+misalign[i],BASE,WIDTH)*amplitudes[i]
    SRF = genSRF(X,IB_wavelengths[i]+misalign[i],BASE,WIDTH)*vary_amp[i]
    SRF_arr[i] = SRF
    plt.plot(X,SRF)
    
    
    band_lower = int(round(i*WIDTH/increment))
    
    band_upper = int(round((i+1)*(WIDTH/increment)))
    
    # Take the mean (Y) value of the SRF within relevant waveband
    IB_val = np.mean(SRF[band_lower:band_upper])
    IB_val_arr[i] = IB_val
    
    ti = tophat(X,0,IB_val,IB_wavelengths[i],WIDTH)
    #plt.plot(X,ti)
    ti_arr[i] = ti

for i in range(N):
    
    # Create a copy of the SRF so we can change it without affect the actual 
    # SRF
    srf_i = copy.deepcopy(SRF_arr[i])
    
    #IB_area[i] = np.trapz(srf_ti[IB_indices],dx=mydx)
    #srf_ti[IB_indices] = srf_ti[IB_indices]/(ti_arr[IB_indices]*WIDTH*increment)
    
    Di = []
    k = 0
    for j in range(0,len(srf_i),step):
        temp_range = srf_i[j:j+step]
        # mean_val = np.mean(temp_range)
        SRF_area = np.trapz(temp_range,dx=increment)
        dij = SRF_area/(IB_val_arr[k]*WIDTH)
        #Di.append(mean_val)
        Di.append(dij)
        k += 1
        
    D[i] = Di

plt.figure(figsize = (10,8))
plt.imshow(D)
plt.colorbar()
plt.title("D")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

plt.figure(figsize = (10,8))
plt.imshow(D,cmap="hot",norm=LogNorm())
plt.colorbar()
plt.show()

print("Condition number k(D) =",np.linalg.cond(D))


# In[5]:


val_check = 39
check_SRF = SRF_arr[val_check]

band_lower = int(round(val_check*WIDTH/increment))

band_upper = int(round((val_check+1)*(WIDTH/increment)))

check_IB_area = np.trapz(check_SRF[band_lower:band_upper],dx=increment)
check_tot_area = np.trapz(check_SRF,dx=increment)

oob_frac = 1 - check_IB_area/check_tot_area
print("Out of band fraction =",oob_frac)

plt.plot(X,check_SRF,'b--')
plt.xlabel("Wavelength ($\lambda$)")
plt.ylabel("Radiance (arb)")
plt.show()


# In[6]:


# plt.figure(figsize=(10,8))
# for i in range(N):
#     plt.plot(X,ti_arr[i]*1/2,)

# plt.plot(X,SRF_arr[0]*0.75,'r--',label="$SRF_1$")
# plt.plot(X,SRF_arr[10],'g--',label="$SRF_j$")
# plt.plot(X,SRF_arr[19]*0.75,'b--',label="$SRF_n$")
# plt.annotate('$a_1$ ', xy=(IB_wavelengths[0], IB_amp*0.75), xytext=(500, IB_amp*0.75),
#              arrowprops=dict(facecolor='black', shrink=0.05),fontsize=18)
# plt.annotate('$a_j$', xy=(IB_wavelengths[10], IB_amp), xytext=(600, IB_amp),
#              arrowprops=dict(facecolor='black', shrink=0.05),fontsize=18)
# plt.annotate('$a_n$', xy=(IB_wavelengths[19], IB_amp*0.75), xytext=(900, IB_amp*0.75),
#              arrowprops=dict(facecolor='black', shrink=0.05),fontsize=18)
# plt.annotate(' ', xy=(IB_wavelengths[4]-WIDTH/2 -3, 407), xytext=(IB_wavelengths[4]+WIDTH/2 +5, 400),
#              arrowprops=dict(arrowstyle="<->",facecolor='black'),)
# plt.text(IB_wavelengths[4]-15,420,"$\Delta \lambda_5$",fontsize=18)
# plt.legend()
# plt.xlabel("Wavelength ($\lambda$)")
# plt.ylabel("Radiance ($W Sr^{-1} \mu m^{-1} m^{-2})$")
# plt.title("Spectral Response Functions and Wavebands")
# #plt.savefig("AnnotatedSRFsWB")
# plt.show()


# In[7]:


plt.figure(figsize=(10,8))
plt.plot(D[10],'b--.')
plt.xlabel("j value")
plt.ylabel("dij")
plt.title("D row (Generated)")
plt.show()


# In[8]:


plt.figure(figsize=(10,8))

for i in range(N):
    plt.plot(X,ti_arr[i])
    
plt.plot(X,SRF_arr[10],'k--')
plt.show()


# In[9]:


# A = D + np.identity(N)

# plt.figure(figsize = (10,8))
# plt.imshow(A)
# plt.colorbar()
# plt.title("A")
# plt.xlabel("j")
# plt.ylabel("i")
# plt.show()

# plt.figure(figsize = (10,8))
# plt.imshow(A,cmap="hot",norm=LogNorm())
# plt.colorbar()
# plt.show()

# print("Condition number k(A) =",np.linalg.cond(A))

print(WIDTH)


# In[10]:


C = np.linalg.inv(D)

plt.figure(figsize = (10,8))
plt.imshow(C)
plt.colorbar()
plt.title("C")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

plt.figure(figsize = (10,8))
plt.imshow(C,cmap="hot",norm=LogNorm())
plt.colorbar()
plt.show()

print("Condition number k(C) =",np.linalg.cond(C))


# In[22]:


Y_meas = []
Y_I = []

for i in range(N):
    y_meas = gen_meas(L,SRF_arr[i])
    Y_meas.append(y_meas)
    
    y_i = gen_meas(L,ti_arr[i])
    Y_I.append(y_i)
    
plt.figure(figsize=(10,8))
# plt.annotate('1', xy=(13, 435000), xytext=(5, 350000),
#              arrowprops=dict(facecolor='black', shrink=0.05),)
# plt.annotate('2', xy=(14, 405000), xytext=(10, 300000),
#              arrowprops=dict(facecolor='black', shrink=0.05),)
# plt.annotate('3', xy=(11, 2.2), xytext=(20, 1.8),
#              arrowprops=dict(facecolor='black', shrink=0.05),)
plt.plot(DN_range,Y_meas)
plt.grid(True)
plt.title("Measurement (Generated)")
plt.xlabel("Waveband")
plt.ylabel("DN")
plt.show()

plt.figure(figsize=(10,8))
plt.plot(X,L,'g',label="L")
plt.plot(X,SRF_arr[13],'b--',label="SRF")
plt.title("SRF[13] and L")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (arb)")
plt.legend()
plt.show()


# In[12]:


# #plt.plot(np.linspace(0,N,len(X)),L,label="L")
# plt.plot(DN_range,Y_meas,label="Measured")
# plt.plot(DN_range,Y_I,label="Ideal in-band")
# plt.legend()
# plt.show()

# diff = 1 - Y_I/Y_meas

# print(diff)


# In[13]:


# noise_scale = np.sqrt(np.mean(Y_meas))*100
# print(noise_scale)

# noise = np.random.randn(N)*noise_scale
# np.save("noise",noise)
noise = np.load("noise.npy")
Y_meas_noise = Y_meas + noise

#Y_meas = Y_meas_noise

Y_clean = np.matmul(C,Y_meas)

plt.plot(DN_range,Y_clean,label="Clean")
plt.plot(DN_range,Y_I,label="Ideal in-band")
plt.legend()
plt.show()


# In[14]:


regen = np.matmul(D,Y_I)

plt.figure(figsize=(10,8))
plt.plot(DN_range,Y_meas,label="Measured")
plt.plot(DN_range,Y_meas_noise,label="Measured with noise")
plt.legend()
plt.title("Measurement with Added Noise")
plt.xlabel("Waveband")
plt.ylabel("DNs")
plt.show()


# In[15]:


Y_meas = np.array(Y_meas)

Y_clean = np.array(Y_clean)

plt.figure(figsize=(10,8))
plt.plot(DN_range,Y_meas,'s-',label="Measured")
plt.plot(DN_range,Y_meas_noise,'rs-',label="Noise")
plt.plot(np.linspace(0,N,len(X)),L,'gs-',label="L")
plt.plot(DN_range,Y_clean,'s-',label="Clean")

plt.legend()
plt.title("Clean comparision with Radiance")
plt.ylabel("Radiance (arb)")
plt.xlabel("Wavelength (arb)")
plt.show()


# In[16]:


# Y_meas_scale = Y_meas/((WIDTH*(dx_scale*100))/(2))
# Y_clean_scale = Y_clean/((WIDTH*(dx_scale*100))/(2))

Y_clean[Y_clean<0] = abs(Y_clean[Y_clean<0])

Y_meas_scale = Y_meas_noise/((IB_val_arr)*(WIDTH))
Y_clean_scale = Y_clean/((IB_val_arr)*(WIDTH))

L_short = shorten_set(L,step)

plt.figure(figsize=(10,8))
plt.plot(DN_range,Y_meas_scale,'s-',label="Measured")
plt.plot(np.linspace(0,N-1,N),L_short,'g-',label="L")
plt.plot(DN_range,Y_clean_scale,'s',label="Clean")
plt.legend()
plt.title("Clean comparision with Radiance")
plt.ylabel("$Radiance (W Sr^{-1} \mu m^{-1} m^{-2})$")
plt.xlabel("Wavelength (arb)")
plt.ylim(0,)
plt.show()


# In[17]:


print(Y_clean[0]/L[0])
print(Y_clean_scale[0]/L[0])
print(WIDTH)


# In[18]:


from scipy.stats import chisquare
from scipy.stats import chi2


plt.figure(figsize=(10,8))
plt.plot(DN_range,L_short,'gs-')
plt.plot(DN_range,Y_clean_scale,'orange')
plt.show()

stat,p = chisquare(Y_clean_scale,L_short)
print("Chistat =",stat)
print("p-value =",p)

another = chi2.cdf(stat,N-1)
another


# In[19]:


mychi = sum((Y_clean_scale - L_short)**2 /L_short)

print(mychi)

