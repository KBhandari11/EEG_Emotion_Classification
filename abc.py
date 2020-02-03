import numpy as np 

raw = np.arange(90).reshape((3,3,2,5))
print(raw)
print(raw.shape)
raw = raw.transpose(1,2,0,3)
print(raw.shape)
raw = raw.reshape(raw, order='F') 
print(raw)
print(raw.shape)
#print(raw.shape)

