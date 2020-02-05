import numpy as np 
data = np.arange(120).reshape((2,3,20))
raw = np.zeros((2,3,4,5))
print("data",data.shape)
for x in range(0,2):
        for y in range(0,3):  
            raw[x,y,:,:]= np.array(np.split(data[x,y,:],4))
print("split",raw.shape)
print(raw) 
raw = np.delete(raw, slice(0,1), axis=2)  
print(raw) 
print("slice",raw.shape)
raw = raw.reshape(2,3,-1)
raw = raw.transpose(1,0,2)
print("transpose: ",raw.shape)
raw = raw.reshape(raw.shape[0],-1, order='F') 
print("sqeeze dimension",raw.shape)
print(raw) 