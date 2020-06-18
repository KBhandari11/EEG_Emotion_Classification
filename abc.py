import numpy as np 

#(40, 19, 60, 128)
def labeling(label_1):
    m = label_1.shape[0] * 2
    label= np.ones((m,2))
    for i in range(label_1.shape[0]):
        for j in range(i*2, i*2+2):
            label[j,0]= 1 if (label_1[i][0] > 50) else 0
            label[j,1]= 1 if (label_1[i][1] > 50) else 0
    return(label) 

a = np.arange(240)
a = np.reshape(a, (120,2))
label = labeling(a)
print(label.shape)
np.savetxt('./label_1.txt', label)