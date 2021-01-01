
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import UnivariateSpline
import copy


def tophat(x, base_level, hat_level, hat_mid, hat_width):
    '''
    Creates data set of tophat function
    
    :param x: data set of spectrum this will be used with
    :param base_level: value of data either side of tophat
    :param hat_level: amplitude of the hat
    :param hat_mid: value of centre of the tophat
    :param hat_width: the width of the tophat
    ''' 
    return np.where((hat_mid-hat_width/2. < x) & (x < hat_mid+hat_width/2.), hat_level, base_level)

def spline_that(x,y,x_new):
#     f = interp1d(x,y)
#     y_new = f(x_new)
    spl = UnivariateSpline(x,y,s=0)
    y_new = spl(x_new)
    
    return y_new


# In[2]:


# Test data with full instrument setup
data = np.load("20170504-response-left.npy")

data = data[:,300:700,:]
data = data[28:78,:,:]
#data = data[:,40:880,:]

# Number of bands
N = 25

# Step size - can be changed
increment = 1e-3

# For converting nm to micro meter
nm_to_um = 1e-3

# Detector Range
Det_range = np.arange(400,1000,increment)

# Shape of datacube
# ACT
X = data.shape[2]
# ALT
Y = data.shape[1]

# Wavelength
Z = data.shape[0]
print("Length of Y =",Y)
print("Length of Z =",Z)

if Z % N != 0:
    raise Exception("Z not evenly divisible by num_bands")

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

# Look at slices of datacube
plt.figure(figsize=(10,8))
plt.imshow(data[Z_slice,:,:])
plt.title("Datacube XY slice at Z = {}".format(Z_slice))
plt.show()

plt.figure(figsize=(10,8))
plt.imshow(data[:,Y_slice,:])
plt.title("Datacube XZ slice at Y = {}".format(Y_slice))
plt.show()

plt.figure(figsize=(10,8))
plt.imshow(data[:,:,X_slice])
plt.title("Datacube YZ slice at X = {}".format(X_slice))
# plt.xlim(0,50)
# plt.ylim(80,111)
plt.show()


# In[3]:


# Centre wavelengths of bands based on user defined N and range
IB_wavelengths = np.linspace(min(Det_range)+WIDTH/2,max(Det_range)-WIDTH/2,N)

# Array for storing SRFs
# N SRFs each Z long
SRF_arr = np.zeros((N,Z))

k = 0
# Gather SRFs as they are in Hypercube
# Gather and save SRFs
for i in range(Y-1,0,-Y_step):
    SRF = data[:,i,X_slice]
    plt.plot(SRF)
    SRF[SRF<0] = 0
    SRF_arr[k] = SRF
    k+=1
print(len(SRF_arr))


# In[4]:

# Array for value of SRF in IB area
IB_val_arr = np.zeros(N)

new_SRF_arr = np.zeros((N,len(Z_new)))

for i in range(N):
    temp_SRF = copy.deepcopy(SRF_arr[i])
    long_SRF = spline_that(Z_range,temp_SRF,Z_new)
    new_SRF_arr[i] = long_SRF
    
    band_lower = int(round(i*WIDTH/increment))
    
    band_upper = int(round((i+1)*(WIDTH/increment)))
    
    IB_val_arr[i] = np.mean(long_SRF[band_lower:band_upper])
    
    
    #ti = tophat(Det_range,0,IB_val,IB_wavelengths[i],WIDTH)
    
D = np.zeros((N,N))    

for i in range(N):
    
    Di = np.zeros(N)
    
    k = 0
    
    SRF_i = new_SRF_arr[i]
    
    for j in range(0,len(long_SRF),new_step):
        # Area in each band
        SRF_area = np.trapz(SRF_i[j:j+new_step],dx=increment)*nm_to_um
        dij = SRF_area/(IB_val_arr[k]*WIDTH*increment)
        
        Di[k] = dij
        k += 1
    
    # Add row to matrix D
    D[i] = Di
    
plt.figure(figsize = (10,8))
plt.imshow(D)
plt.colorbar()
plt.show()
plt.figure(figsize = (10,8))
plt.imshow(D,cmap="hot",norm=LogNorm())
plt.colorbar()
plt.show()

print(np.linalg.cond(D))


# In[5]:

# Invert D
C = np.linalg.inv(D)

plt.figure(figsize = (10,8))
plt.imshow(C)
plt.colorbar()
plt.show()
plt.figure(figsize = (10,8))
plt.imshow(C,cmap="hot",norm=LogNorm())
plt.colorbar()
plt.show()

print(np.linalg.cond(C))


# In[6]:


def step(x):
    '''
    Step function 
    Returns 1 for first half of x and 0 for second half
    '''
    return 929 * (x < x[int(len(x)/2)])

#L = step(Z_new)

file = 'radiances.txt'
        
f = open(file)

rad_data = f.read().split("\n")
rad_data = rad_data[:-1]

wavelength = []
r1 = []
r2 = []
r3 = []
r4 = []

for lines in rad_data:
    numbers = lines.split(" ")
    wavelength.append(float(numbers[0]))
    r1.append(float(numbers[1]))
    r2.append(float(numbers[2]))
    r3.append(float(numbers[3]))
    r4.append(float(numbers[4]))
    
r1 = r3[:-1]
r1 = spline_that(np.arange(0,600,1),r1,np.arange(0,600,increment))
L = r1

len_radiances = len(r1)
radiance_step = round(len_radiances/N)

plt.figure(figsize=(10,8))
plt.plot(r1,label="L")
#plt.plot(ALT,FP_trans_show,label="Ideal Filter Transmission")
# plt.plot(wavelength,r2,label="water")
# plt.plot(wavelength,r3,label="vegetation")
# plt.plot(wavelength,r4,label="soil")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance")
plt.legend()
plt.show()


plt.figure(figsize=(10,8))

for i in range(N):
    plt.plot(SRF_arr[i])

plt.plot(np.linspace(0,50,len(Z_new)),L)
plt.show()


# In[7]:


def gen_meas(L,SRF):
    product = L*SRF
    
    area = np.trapz(product,dx=increment)*nm_to_um
    
    return area


# In[8]:

# Create measurement from SRFs and radiance
Y_meas = np.zeros(N)

for i in range(N):
    long_SRF = spline_that(Z_range,SRF_arr[i],Z_new)
    
    temp_area = np.zeros(N)
    k = 0
    for j in range(0,len(long_SRF),new_step):
        # Area of SRF*L within each band
        band_j = gen_meas(L[j:j+new_step],long_SRF[j:j+new_step])
        temp_area[k] = band_j
        k += 1
        
    # Each band reading is the total area for that SRF
    Y_meas[i] = sum(temp_area)
    
    
plt.plot(Y_meas)
plt.show()


# In[9]:

# Clean measurement
clean = np.matmul(C,Y_meas)

plt.plot(clean)
plt.show()


# In[10]:

# Scaling
meas_scale = Y_meas/(IB_val_arr*WIDTH*increment)
clean_scale = clean/(IB_val_arr*WIDTH*increment)

plt.figure(figsize=(10,8))
plt.plot(DN_range,meas_scale,'s-',label="Measured")
plt.plot(DN_range,clean_scale,'s-',label="Clean")
plt.plot(np.linspace(0,N-1,len(L)),L,'-',label="L")
plt.legend()
plt.show()

