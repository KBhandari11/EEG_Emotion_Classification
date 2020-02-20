import numpy as np 
from scipy import signal


data = np.arange(120).reshape((2,3,20))
raw = np.zeros((2,3,4,5))
print("data",data)
for x in range(0,2):
        for y in range(0,3):  
            raw[x,y,:,:]= np.array(np.split(data[x,y,:],4))
print("split",raw.shape)
print("-----------------")
raw = np.delete(raw, slice(0,2), axis=2)  
print("-----------------")
print("slice",raw.shape)

"""
raw = raw.reshape(2,3,-1)
print("-----------------")
raw = raw.transpose(1,0,2)
print("transpose: ",raw.shape)
raw = raw.reshape(raw.shape[0],-1) 
print("sqeeze dimension",raw.shape)
print("-----------------")
freq = np.zeros((3,2,10))
for x in range(0,3):
    for y in range(0, 2):
            freq[x,:]= np.array(np.split(raw[x,:],2)) 
print("split",freq.shape)
print("-----------------")
"""
print(raw)
raw = raw.reshape(raw.shape[0],-2)       # Reduce the dimension to 2x2 array (19, 307200)
print(raw)
print(raw.shape)